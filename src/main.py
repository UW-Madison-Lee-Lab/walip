import os, torch, sys
import numpy as np
import torch.nn.functional as F
from scipy import stats
from tclip.clip_ops import evaluate_classification, evaluate_multilabel_classification
from utils.text_loader import get_word2id, load_vocabs, load_vocabs_from_pairs
from utils.helper import dict2clsattr
from translation import align_words, load_test_dict, train_supervision, cal_similarity
from models.embedding import ClipEmbedding
import argparse, json
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--config", type=str, default='translation')
args = parser.parse_args()
with open(f"configs/{args.config}.json") as f:
    model_config = json.load(f)
params = dict2clsattr(vars(args), model_config)
params.emb_dir = os.path.join(params.dict_pth, 'embeddings/')
params.img_dir = os.path.join(params.dict_pth, f'images/{params.image_data}/')
params.txt_dir = os.path.join(params.dict_pth, f'texts/{params.word_data}/')
params.langs = {'src': params.src_lang, 'tgt': params.tgt_lang}
params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


################# ================= ################
def prepare_embeddings(emb_type, txt_dir, word_data, data_mode, opts):
    vocabs, word2ids, embs = {}, {}, {}
    if word_data == 'wiki':
        vocabs = load_vocabs_from_pairs(opts)
    for l in ['src', 'tgt']:
        if not (word_data == 'wiki'):
            vocabs[l] = load_vocabs(opts.langs[l], txt_dir, word_data, data_mode)
        word2ids[l] = get_word2id(vocabs[l])
        embs[l] = ClipEmbedding(emb_type, opts.langs[l], data_mode, opts).load_embedding(vocabs[l])
        embs[l] = torch.from_numpy(embs[l]).to(opts.device)
    return word2ids, embs


###============= Working Mode =============##########
if params.work_mode == 'analysis':
    # evaluate_classification(params.image_data, params.src_lang, params)
    evaluate_classification(params.image_data, params.tgt_lang, params)
    # evaluate_multilabel_classification(params.image_data, params.tgt_lang, params)
    sys.exit("DONE!!!!") 

if params.work_mode == 'preprocess':
    from utils.filter_images import find_correct_images, find_interesection
    find_correct_images(params.src_lang, params)
    find_correct_images(params.tgt_lang, params)
    find_interesection(params.image_data, params)
    sys.exit("DONE!!!!") 

if params.work_mode == 'image_retrieval':
    l = 'src'
    if params.word_data == 'wiki':
        vocab_pairs = load_vocabs_from_pairs(params)
        vocabs = vocab_pairs[l]
    else:
        vocabs = load_vocabs(params, params.langs[l])
    embObj = ClipEmbedding(params.emb_type, params.langs[l], params.data_mode, params)
    txt_embs = embObj.load_clip_txt_emb(vocabs)
    img_embs = embObj.load_clip_img_emb().cpu()
    # normed_img_embs = F.normalize(img_embs, dim=-1)
    # normed_txt_embs = F.normalize(torch.Tensor(txt_embs), dim=-1)
    u = img_embs[470:480, :] @ txt_embs.T * 10
    u = u.softmax(dim=-1)
    s = u.argsort(descending=True, dim=1)
    print(s[:, :3])
    from IPython import embed; embed()
    sys.exit("DONE!!!!") 

if params.work_mode == 'retrieval':
    vocabs, sims, indices = {}, {}, {}
    for l in ['src', 'tgt']:
        vocabs[l] = load_vocabs(params, params.langs[l])
        embObj = ClipEmbedding(params.emb_type, params.langs[l], params.data_mode, params)
        txt_embs = embObj.load_clip_txt_emb(vocabs[l])
        img_embs = embObj.load_clip_img_emb().cpu()
        # normed_img_embs = F.normalize(img_embs, dim=-1)
        # normed_txt_embs = F.normalize(torch.Tensor(txt_embs), dim=-1)
        u = img_embs @ txt_embs.T * 10
        u = u.softmax(dim=-1)
        indices[l] = u.argsort(descending=True, dim=1)[:, :5]
        sims[l] = u
        
    # looks at all words embedding: -> find the closest 
    N = img_embs.shape[0]
    count = 0
    total = 0
    # cifar100
    for i in range(N//params.num_images):
        k = i * params.num_images
        ind = {}
        for l in ['src', 'tgt']:
            inds = indices[l][k: k + 10, :].numpy()
            ind[l] = stats.mode(inds[:, 0])
            # repeat words
        # a, b = indices['src'][i], indices['tgt'][i]
        # sim_a, sim_b = sims['src'][i][a], sims['tgt'][i][b],
        # print( a, b, sim_a, sim_b, vocabs['src'][a], vocabs['tgt'][b])
        if ind['src'][1][0] > 3 and ind['tgt'][1][0] > 3:
            total += 1
        else:
            continue
        if ind['src'][0][0] == ind['tgt'][0][0]:
            count+= 1
        else:
            print(i, vocabs['src'][ind['src'][0][0]], vocabs['tgt'][ind['tgt'][0][0]])
    print('Correctness {:.4f} - Recall: {:.4f}'.format(count/total, count/N*params.num_images))
    from IPython import embed; embed()
    sys.exit("DONE!!!!") 

if params.work_mode == 'translation': # translation
    if params.method == "semi":
        word2ids, embs = prepare_embeddings(params.emb_type, params.txt_dir, params.word_data, params.data_mode, params)
        test_dico = load_test_dict(params, word2ids)
        inds = np.load(f'../results/indices_{params.src_lang}_{params.tgt_lang}_{params.word_data}.npy', allow_pickle=True)
        X0 = embs['src'][inds[:, 0], :]
        X1 = embs['tgt'][inds[:, 1], :]
        W = train_supervision(X0, X1)

        params.word_data = 'wiki'
        params.txt_dir = os.path.join(params.dict_pth, f'texts/{params.word_data}/')
        word2ids, embs = prepare_embeddings(params.emb_type, params.txt_dir, params.word_data, params.data_mode, params)
        test_dico = load_test_dict(params, word2ids)
        for _ in range(15):  # refinement
            embs['src'] = embs['src'] @ W.T
            scores = cal_similarity(params.sim_score, test_dico, embs) 
            inds = align_words(params.matching_method, test_dico, scores)
            X0 = embs['src'][inds[:, 0], :]
            X1 = embs['tgt'][inds[:, 1], :]
            W = train_supervision(X0, X1)

        embs['src'] = embs['src'] @ W.T
        
    elif params.method == 'supervised':
        # training
        word2ids, embs = prepare_embeddings(params.emb_type, params.txt_dir, params.word_data, params.data_mode, params)
        W = train_supervision(embs['src'], embs['tgt'])
        params.word_data = 'wiki'
        params.txt_dir = os.path.join(params.dict_pth, f'texts/{params.word_data}/')
        word2ids, embs = prepare_embeddings(params.data_mode)
        embs['src'] = embs['src'] @ W.T
    elif params.method == 'unsupervised':
        word2ids, embs = prepare_embeddings(params.emb_type, params.txt_dir, params.word_data, params.data_mode, params)

    ##### Testing pairs  (subsets of dictionaries)
    embs['src']= embs['src'].type(torch.FloatTensor).to(params.device)
    test_dico = load_test_dict(params, word2ids)
    print('\n..... Evaluating ..... ', params.word_data, params.emb_type, params.sim_score, params.matching_method)    
    scores = cal_similarity(params.sim_score, test_dico, embs) 
    lst = align_words(params.matching_method, test_dico, scores)
    # np.save(f'../results/indices_{params.src_lang}_{params.tgt_lang}_{params.word_data}', lst)

    ###============= Translation =============##########
    # decorrelate
    # if params.emb_type == 'cliptext':
    #     W = scipy.stats.ortho_group.rvs(embs['src'].shape[1])
    #     W = torch.from_numpy(W).type(torch.FloatTensor).cuda()
    #     embs['src'] = embs['src'] @ W.T
    sys.exit("DONE!!!!") 
