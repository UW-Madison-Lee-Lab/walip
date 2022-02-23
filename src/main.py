import os, torch, sys
import numpy as np
import torch.nn.functional as F
from scipy import stats
from tclip.clip_ops import zero_shot_classification
from utils.text_loader import get_word2id, load_vocabs, load_vocabs_from_pairs
from utils.helper import dict2clsattr
from translation import align_words, load_test_dict, train_supervision, cal_similarity, robust_procrustes
from models.embedding import ClipEmbedding
from models.ops import load_models
import argparse, json
import configs
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--save_lst", default=0, type=int)
parser.add_argument("-c", "--config", type=str, default='u')
args = parser.parse_args()
config_file = {
    'u': 'unsupervised',
    's': 'semi',
    'a': 'analysis'
}[args.config]
with open(f"configs/{config_file}.json") as f:
    model_config = json.load(f)
for s in ['reuse_embedding', 'reuse_text_embedding', 'reuse_image_embedding', 'reuse_image_data']:
    model_config["work"][s] = model_config["work"][s] == 1
args = dict2clsattr(vars(args), model_config)
args.langs = {'src': args.src_lang, 'tgt': args.tgt_lang}
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################# ================= ################
def prepare_embeddings(emb_type, langs, word_data, data_mode, opts, multi_embs=False):
    word2ids, embs = {}, {}
    for key, lang in langs.items():
        vocab = load_vocabs(lang, langs, word_data, data_mode)
        embObj = ClipEmbedding(emb_type, lang, data_mode, opts)
        if multi_embs:
            txt_embs = embObj.load_clip_txt_emb(vocab)
            img_embs = embObj.load_clip_img_emb()
            embs[key] = {'img': img_embs, 'txt': txt_embs}
        else:
            embs[key] = torch.from_numpy(embObj.load_embedding(vocab)).to(opts.device)
        word2ids[key] = get_word2id(vocab)
    return word2ids, embs


###============= Working Mode =============##########
if args.work_mode == 'analysis':
    lang = args.tgt_lang
    model, logit_scale, preprocess = load_models(lang, configs.model_names[lang], 'coco', args.device, args.large_model) 
    zero_shot_classification(model, args.image_data, args.word_data, lang, args.num_prompts, logit_scale, preprocess, device=args.device, test_perclass=args.test_perclass)
    sys.exit("DONE!!!!") 

if args.work_mode == 'preprocess':
    from utils.filter_images import find_correct_images, find_interesection
    find_correct_images(args.src_lang, args)
    find_correct_images(args.tgt_lang, args)
    find_interesection(args.image_data, args)
    sys.exit("DONE!!!!") 

if args.work_mode == 'image_retrieval':
    l = 'src'
    _, embs = prepare_embeddings(args.emb_type, {l: args.langs[l]}, args.word_data, args.data_mode, args, multi_embs=True)
    txt_embs, img_embs = embs['txt'], embs['img'].cpu()
    u = img_embs[470:480, :] @ txt_embs.T * 10
    u = u.softmax(dim=-1)
    s = u.argsort(descending=True, dim=1)
    print(s[:, :3])
    sys.exit("DONE!!!!") 

if args.work_mode == 'retrieval':
    word2ids, embs = prepare_embeddings('fp', args.langs, args.word_data, args.data_mode, args, multi_embs=False)
    top2 = {}
    medians = {}
    for l in ['src', 'tgt']:
        # fp is word @ images 
        top2[l] = torch.topk(embs[l], 2, dim=0)
        # top2[l] = torch.max(embs[l], dim=0)
        medians[l] = (top2[l][0][0, :] - top2[l][0][1, :]).median()
    
    def is_good(top2, i, c=0):
        x1, x2 = top2['src'][0][:, i]
        y1, y2 = top2['tgt'][0][:, i]
        if x1 - x2 > medians['src'] and  y1-y2 > medians['tgt'] and x1 > 0 and y1 > 0:
            return True
        return False

    # looks at all words embedding: -> find the closest 
    N = embs['src'].shape[1]
    count, total = 0, 0
    wrong_pairs = []
    # inds = indices[l][k: k + 10, :].numpy()
    # ind[l] = stats.mode(inds[:, 0])
    for i in range(N):
        if is_good(top2, i, c=0):
            total += 1
            x, y = top2['src'][1][0, i].item(), top2['tgt'][1][0, i].item()
            if x == y:
                count+= 1
            else:
                wrong_pairs.append([i, x, y])
    # print('wrong-pairs ', wrong_pairs)
    print('Retrieval: {}/{} -- Correct {:.2f}/100 - Recall: {:.2f}/100'.format(count, total, count/total*100, count/N * 100))
    from IPython import embed; embed()
    sys.exit("DONE!!!!") 

if args.work_mode == 'retrieval_k':
    word2ids, embs = prepare_embeddings('fp', args.langs, args.word_data, args.data_mode, args, multi_embs=False)
    K = args.num_images
    top2 = {}
    medians = {}
    for l in ['src', 'tgt']:
        # fp is word @ images 
        top2[l] = torch.topk(embs[l], 2, dim=0)
        # top2[l] = torch.max(embs[l], dim=0)
        medians[l] = (top2[l][0][0, :] - top2[l][0][1, :]).median()

    # looks at all words embedding: -> find the closest 
    N = embs['src'].shape[1]
    count, total = 0, 0
    wrong_pairs, correct_pairs = [], []
    # inds = indices[l][k: k + 10, :].numpy()
    # ind[l] = stats.mode(inds[:, 0])
    for i in range(N//K):
        m = i * K
        x = stats.mode(top2['src'][1][0, m:m+K].cpu().numpy().flatten())
        y = stats.mode(top2['tgt'][1][0, m:m+K].cpu().numpy().flatten())
        if x[1][0] >= 3 and y[1][0] >= 3:
            total += 1
            if x[0][0] == y[0][0]:
                count+= 1
                correct_pairs.append([x[0][0], y[0][0]])
            else:
                wrong_pairs.append([i, x[0][0], y[0][0]])
    # print('wrong-pairs ', wrong_pairs)
    print('Retrieval: {}/{} -- Correct {:.2f}/100 - Recall: {:.2f}/100'.format(count, total, count/total*100, count/N * K * 100))
    from IPython import embed; embed()
    sys.exit("DONE!!!!") 

if args.work_mode == 'translation': # translation
    lst_path = f'../results/indices_{args.src_lang}_{args.tgt_lang}_{args.word_data}_{args.large_model}.npy'
    if args.method == "semi":
        word2ids, embs = prepare_embeddings(args.emb_type, args.langs, args.word_data, args.data_mode, args)
        test_dico = load_test_dict(args, word2ids)
        inds = np.load(lst_path, allow_pickle=True)
        X0 = embs['src'][inds[:, 0], :]
        X1 = embs['tgt'][inds[:, 1], :]
        # X0 = embs['src'][:500, :]
        # X1 = embs['tgt'][:500, :]
        W = train_supervision(X0, X1)

        # args.word_data = 'wiki'
        # word2ids, embs = prepare_embeddings(args.emb_type, args.langs, args.word_data, args.data_mode, args)
        # test_dico = load_test_dict(args, word2ids)
        for _ in range(15):  # refinement
            embs['src'] = embs['src'] @ W.T
            scores = cal_similarity(args.sim_score, test_dico, embs, args.row_ranking) 
            inds = align_words(args.matching_method, test_dico, scores)
            # from IPython import embed; embed()
            X0 = embs['src'][inds[:, 0], :]
            X1 = embs['tgt'][inds[:, 1], :]
            W = train_supervision(X0, X1)

        embs['src'] = embs['src'] @ W.T
        
    elif args.method == 'supervised':
        # training
        word2ids, embs = prepare_embeddings(args.emb_type, args.langs, args.word_data, args.data_mode, args)
        W = train_supervision(embs['src'], embs['tgt'])
        # args.word_data = 'wiki'
        # word2ids, embs = prepare_embeddings(args.emb_type, args.langs, args.word_data, args.data_mode, args)
        embs['src'] = embs['src'] @ W.T
    elif args.method == 'unsupervised':
        word2ids, embs = prepare_embeddings(args.emb_type, args.langs, args.word_data, args.data_mode, args)

    ##### Testing pairs  (subsets of dictionaries)
    embs['src']= embs['src'].type(torch.FloatTensor).to(args.device)
    test_dico = load_test_dict(args, word2ids)
    print('\n..... Evaluating ..... ', args.word_data, args.emb_type, args.sim_score, args.matching_method)    
    scores = cal_similarity(args.sim_score, test_dico, embs, args.row_ranking) 
    lst = align_words(args.matching_method, test_dico, scores)
    if args.save_lst:
        np.save(lst_path, lst)

    ###============= Translation =============##########
    # decorrelate
    # if args.emb_type == 'cliptext':
    #     W = scipy.stats.ortho_group.rvs(embs['src'].shape[1])
    #     W = torch.from_numpy(W).type(torch.FloatTensor).cuda()
    #     embs['src'] = embs['src'] @ W.T
    sys.exit("DONE!!!!") 
