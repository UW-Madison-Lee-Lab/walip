import os, torch
import numpy as np
import torch.nn.functional as F
from scipy import stats
from tclip.clip_ops import evaluate_classification, evaluate_multilabel_classification
from utils.text_loader import get_word2id, load_vocabs, load_vocabs_from_pairs
from utils.helper import dict2clsattr
from translation import evaluate_translation, match_embeddings, train_supervision
from models.embedding import ClipEmbedding
import argparse, json
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)
# parse parameters
args = parser.parse_args()
with open("translation_configs.json") as f:
    model_config = json.load(f)
train_config = vars(args)
params = dict2clsattr(train_config, model_config)
params.emb_dir = params.dict_pth + 'embeddings/'
params.img_dir = params.dict_pth + f'images/{params.image_data}/'
params.txt_dir = params.dict_pth + f'texts/{params.word_data}/'
params.langs = {'src': params.src_lang, 'tgt': params.tgt_lang}
params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################# ================= ################

def prepare_embeddings(data_mode):
    vocabs, word2ids, embs = {}, {}, {}
    if params.word_data == 'wiki':
        vocabs = load_vocabs_from_pairs(params)
    for l in ['src', 'tgt']:
        if not (params.word_data == 'wiki'):
            vocabs[l] = load_vocabs(params, params.langs[l])
        word2ids[l] = get_word2id(vocabs[l])
        embs[l] = ClipEmbedding(params.emb_type, params.langs[l], data_mode, params).load_embedding(vocabs[l])
        embs[l] = torch.from_numpy(embs[l]).to(params.device)
        # embs[l] = F.normalize(embs[l], dim=1)
    return word2ids, embs


###============= Working =============##########

if params.work_mode == 'analysis':
    evaluate_classification(params.image_data, params.src_lang, params)
    # evaluate_classification(params.image_data, params.tgt_lang, params)
    # evaluate_multiclass_classification(params.image_data, params.tgt_lang, params)
elif params.work_mode == 'preprocess':
    from utils.filter_images import find_correct_images, find_interesection
    find_correct_images(params.src_lang, params)
    find_correct_images(params.tgt_lang, params)
    find_interesection(params.image_data, params)
elif params.work_mode == 'image_retrieval':
    l = 'src'
    if params.word_data == 'wiki':
        vocab_pairs = load_vocabs_from_pairs(params)
        vocabs = vocab_pairs[l]
    else:
        vocabs = load_vocabs(params, params.langs[l])
    embObj = ClipEmbedding(params.emb_type, params.langs[l], params.data_mode, params)
    txt_embs = embObj.load_clip_txt_emb(vocabs)
    img_embs = embObj.load_clip_img_emb().cpu()
    normed_img_embs = F.normalize(img_embs, dim=-1)
    normed_txt_embs = F.normalize(torch.Tensor(txt_embs), dim=-1)
    u = normed_img_embs[80:90, :] @ normed_txt_embs.T * 10
    u = u.softmax(dim=-1)
    s = u.argsort(descending=True, dim=1)
    print(s[:, :3])

elif params.work_mode == 'retrieval':
    from scipy.special import softmax
    # give an image 
    # load image embeddings
    vocabs, sims, indices = {}, {}, {}
    if params.word_data == 'wiki':
        vocabs = load_vocabs_from_pairs(params)
    for l in ['src', 'tgt']:
        if not (params.word_data == 'wiki'):
            vocabs[l] = load_vocabs(params, params.langs[l])
        embObj = ClipEmbedding(params.emb_type, params.langs[l], params.data_mode, params)
        txt_embs = embObj.load_clip_txt_emb(vocabs[l])
        img_embs = embObj.load_clip_img_emb().cpu()
        normed_img_embs = F.normalize(img_embs, dim=-1)
        normed_txt_embs = F.normalize(torch.Tensor(txt_embs), dim=-1)
        u = normed_img_embs @ normed_txt_embs.T * 10
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
        if ind['src'][1][0] > 4 and ind['tgt'][1][0] > 4:
            total += 1
        else:
            continue
        print(i, vocabs['src'][ind['src'][0][0]], vocabs['tgt'][ind['tgt'][0][0]])
        if ind['src'][0][0] == ind['tgt'][0][0]:
            count+= 1
            print('Correct --')
    print('Correctness {:.4f} - Recall: {:.4f}'.format(count/total, count/N*params.num_images))
    from IPython import embed; embed()
else: # translation
    print("..... Prepare embeddings ..... ")
    if params.supervised:
        # training
        word2ids, embs_train = prepare_embeddings('test')
        # test
        # get the training set:
        # test_id1 = dico[:, 0]
        # train_id1 = []
        # for i in range(len(word2ids['src'])):
        #     if not (i in test_id1):
        #         train_id1.append(i)
        # print("Len of train set: ", len(train_id1))
        # embs_train0 = embs['src'][train_id1, :]
        # embs_train1 = embs['tgt'][train_id1, :]
        W = train_supervision(embs_train['src'], embs_train['tgt'])
        # test(word2ids, {'src': embs_train['src']@W.T, 'tgt': embs_train['tgt']})
        # for i in range(5):
        #     W = refinement(W, embs_train)
        #     test(word2ids, {'src': embs_train['src']@W.T, 'tgt': embs_train['tgt']})
        # evaluate test
        # W = np.load('../dicts/best_W.npy', allow_pickle=True)
        # W = torch.Tensor(W).cuda()
        params.word_data = 'noun'
        params.txt_dir = params.dict_pth + f'texts/{params.word_data}/'
        word2ids, embs = prepare_embeddings(params.data_mode)
        embs['src'] = embs['src'] @ W.T
    else:
        ##### Vocabularies
        word2ids, embs = prepare_embeddings(params.data_mode)
    embs['src']= embs['src'].type(torch.FloatTensor).to(params.device)
    # test(word2ids, {'src': embs['tgt'], 'tgt': embs['tgt']})
    if params.sim_score == 'ranking':
        match_embeddings(params, None, embs)
    else:
        evaluate_translation(word2ids, embs)
    ###============= Translation =============##########
    # decorrelate
    # if params.emb_type == 'cliptext':
    #     W = scipy.stats.ortho_group.rvs(embs['src'].shape[1])
    #     W = torch.from_numpy(W).type(torch.FloatTensor).cuda()
    #     embs['src'] = embs['src'] @ W.T