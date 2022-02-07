import os, torch
import numpy as np

import scipy
import scipy.stats
import scipy.optimize
from scipy import stats
from pkg.gmp import quadratic_assignment_ot

import torch.nn.functional as F

from evals.word_translation import get_csls_word_translation, build_dictionary, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy
from utils.text_loader import get_word2id, load_vocabs, combine_files, load_vocabs_from_pairs
from models.embedding import ClipEmbedding
from tclip.clip_ops import evaluate_classification, evaluate_multilabel_classification
import configs
import argparse
import json 
from utils.helper import dict2clsattr

os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
# parser.add_argument("-w", "--word_data", type=str, default="cifar100", help="Dataset for word translation")
# parser.add_argument("-i", "--image_data", type=str, default="cifar100", help="Image dataset for fingerprint")
# parser.add_argument("-l", "--learning_mode", type=str, default="unsupervised", \
#     help="learning method: unsupervised or supervised")
# parser.add_argument("-m", "--matching_method", type=str, default="lin_asm",\
#      help="biparite matching method [nn, lin_asm]")
# parser.add_argument("-s", "--sim_score", type=str, default="cosine", \
#     help="similarity score [cosine, csls, inner_prod]")
# parser.add_argument("-t", "--data_mode", type=str, default="test", help="mode of evaluation")
# parser.add_argument("-e", "--emb_type", type=str, default="fp", \
#     help="type of embedding: fingerprint, cliptext, fasttext")
# data
# parser.add_argument("--src_lang", type=str, default='en', help="Source language")
# parser.add_argument("--tgt_lang", type=str, default='sw', help="Target language")
# parser.add_argument("-large_model", action='store_true', help="Analysis mode")
# parser.add_argument("--work_mode", type=str, default='translation', help="Analysis mode")

# parser.add_argument("--num_images", type=int, default=1, help="Number of imager per class")
# parser.add_argument("--num_prompts", type=int, default=1, help="Number of text prompts")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=2)

# parser.add_argument("-supervised", action='store_true', help="")
# parser.add_argument("-reuse_embedding", action='store_true', help="")
# parser.add_argument("-reuse_image_embedding", action='store_true', help="")
# parser.add_argument("-reuse_image_data", action='store_true', help="")
# parser.add_argument("-using_filtered_images", action='store_true', help="")
# parser.add_argument("-reuse_text_embedding", action='store_true', help="")

parser.add_argument("--dict_pth", type=str, default="../dicts/", help="paths to dicts")

# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")


# parse parameters
args = parser.parse_args()

config_path = "translation_configs.json"
with open(config_path) as f:
    model_config = json.load(f)
train_config = vars(args)
params = dict2clsattr(train_config, model_config)
params.emb_dir = params.dict_pth + 'embeddings/'
params.img_dir = params.dict_pth + f'images/{params.image_data}/'
params.txt_dir = params.dict_pth + f'texts/{params.word_data}/'
params.langs = {'src': params.src_lang, 'tgt': params.tgt_lang}
params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################# ================= #######################

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

def quad_ot(embs):
    correct_inds = np.arange(len(embs['src']))
    def compute_match_ratio(inds, correct_inds):
        matched = inds == correct_inds
        return np.mean(matched)
    goat_options = dict(maximize=True, maxiter=150, tol=1e-5, shuffle_input=True)
    n_init = 3
    N = len(embs['src'])
    if True:
        ems = []
        X = {}
        for s in ['src', 'tgt']:
            X[s] = embs[s][:N, ...]
            e = X[s].cpu().numpy()
            a = e @ e.T
            indices = np.argsort(a, axis=1)
            ind = np.zeros(indices.shape)
            for k in range(N):
                for i, j in enumerate(indices[k, ...][::-1]):
                    ind[k][j] = i
            ems.append(ind)
            # ems.append(e @ e.T)
            # for i in range(len(embs[s])):
            #     embs[s][i][i] = 0
        
        row = None
        rows = []
        best_match = 0
        for i in range(n_init):
            print(i)
            # res = quadratic_assignment(male_adj, herm_adj, options=vanilla_options)
            # for reg in [100, 300, 500, 700]:  # above 700 usually breaks
            reg = 100
            goat_options["reg"] = reg
            res = quadratic_assignment_ot(ems[0], ems[1], options=goat_options)
            # res["match_ratio"] = compute_match_ratio(res["col_ind"], correct_inds)
            # res["method"] = "goat"
            # res["reg"] = reg
            rows.append(res["col_ind"])
            # print(res["match_ratio"] > best_match, i, reg, res["match_ratio"])
            # if res["match_ratio"] > best_match:
                # best_match = res["match_ratio"]
                # row = res["col_ind"]
        
        # print(f"{time.time() - currtime:.3f} seconds elapsed.")
        # print("Best: {:.2f}".format(best_match))
        # supervision
        rows = np.stack(rows, axis=0)
        m = stats.mode(rows)
        row = m[0][0].tolist()
        match_ratio = compute_match_ratio(row, correct_inds)
        print(match_ratio)
        # print('Supervision')
        # X['tgt'] = X['tgt'][row]
        # W = train_supervision(X1=X['src'], X2=X['tgt'])
        # # embs['tgt'][:N, ...] = X['tgt'] ### why? CSLS
        # embs['src'] = embs['src'] @ W.T
        # scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], 'csls')
        # results = get_topk_translation_accuracy(dico, scores)
        # print(results)
    else:
        # remove [i, i]?
        goat_options["reg"] = 100
        def get_ot(X, Y):
            A = X @ X.T
            F = np.eye(Y.shape[1])
            Z = np.zeros((X.shape[0], Y.shape[0]))
            for i in range(10):
                Yf = X @ F
                B = Yf @ Yf.T
                res = quadratic_assignment_ot(A, B, options=goat_options)
                res["match_ratio"] = compute_match_ratio(res["col_ind"], correct_inds)
                print(i, "Match: {:.2f}".format(res["match_ratio"]))

                inds = res['col_ind']
                P = Z.copy()
                for k in range(X.shape[0]):
                    P[k, inds[k]] = 1
                Q, Sig, QT = scipy.linalg.svd(P @ (A @ P.T), full_matrices=True)
                D = Z.copy()
                for k in range(len(Sig)):
                    D[k, k] = Sig[k]
                F = Y.T @ np.linalg.inv(Y@Y.T) @ Q @ (D**0.5)   
                print(F == np.eye(Y.shape[1]))

        for s in ['src', 'tgt']:
            embs[s] = embs[s].cpu().numpy()
        get_ot(embs['src'], embs['tgt'])

def get_similarity_score(dico, embs):
    if params.sim_score in ['csls', 'cosine']:
        scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], params.sim_score)
    elif params.sim_score == 'inner_prod':
        test_emb0 = embs['src'][dico[:, 0]]
        test_emb1 = embs['tgt'][dico[:, 1]]
        scores = test_emb0 @ test_emb1.T 
    elif params.sim_score == 'ranking':
        ranks = {}
        for l in ['src', 'tgt']:
            ranks[l] = torch.topk(embs[l], 10, dim=1)[1].cpu().numpy()

        def similarity(a, b):
            s = 0
            for i in range(10):
                if a[i] in b:
                    j = np.where(b == a[i])[0]
                    s += (10 - i) * (10 - j)
            return s

        N = len(ranks['src'])
        scores = np.zeros((N, N))
        for k in range(N):
            for m in range(N):
                scores[k, m] = similarity(ranks['src'][k, :], ranks['tgt'][m, :])
        col_ind = scores.argmax(axis=1)
        # filter 
        threshold = 200
        count = 0
        total = 0
        for i in range(len(scores)):
            if scores[i][col_ind[i]] > threshold:
                total += 1
                if col_ind[i] == i:
                    count += 1
                    
        print(count, total, count/total*100)
        from IPython import embed; embed()
    
    return scores

def eval(dico, embs):
    print('\n..... Evaluating ..... ', params.word_data, params.emb_type, params.sim_score, params.matching_method)        

    ###============= Similarity Calculation =============##########
    # from IPython import embed; embed()
    scores = get_similarity_score(dico, embs)

    ###============= Matching Algorithm =============##########
    if params.matching_method == 'nn':
        results = get_topk_translation_accuracy(dico, scores)
        print(results)
        lstp = []
        s = scores.max(dim=1)
        correct, total = 0, 0
        for i in range(len(scores)):
            if s[0][i] > 0:
                lstp.append(i)
                total +=1
                if s[1][i] == i:
                    correct += 1
        print("Precision@1 ", correct, total, correct/total)
        from IPython import embed; embed()
    elif params.matching_method == 'hungarian':
        cost = -scores.cpu().numpy()
        dico = dico.cpu()    
        _, col_ind = scipy.optimize.linear_sum_assignment(cost)
        acc, wrong_pairs = get_accuracy(dico, col_ind)
    elif params.matching_method == 'quad_ot':
        quad_ot(embs)
        
def test(word2ids, embs):
    ##### Testing pairs  (subsets of dictionaries)
    test_fpath = params.txt_dir + f'{params.word_data}_{params.src_lang}_{params.tgt_lang}_{params.data_mode}.txt'
    if not os.path.isfile(test_fpath):
        combine_files(params)
    test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'], delimiter=configs.delimiters[params.word_data])
    test_dico = test_dico.cuda()
    eval(test_dico, embs)

###============= Supervision =============##########
def train_supervision(X1, X2):
    # X = X1.T @ X2
    # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
    M = X2.transpose(0, 1).mm(X1).cpu().numpy()
    U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
    # W = U @ VT
    W = U.dot(VT)
    return torch.Tensor(W).cuda()

def refinement(W, embs):
    # Get the best mapping according to VALIDATION_METRIC
    # trainer.reload_best()

    # training loop
    # build a dictionary from aligned embeddings
    src_emb = embs['src'] @ W.T
    tgt_emb = embs['tgt']
    src_emb = F.normalize(src_emb, dim=-1)
    tgt_emb = F.normalize(tgt_emb, dim=-1)
    
    dico = build_dictionary(src_emb, tgt_emb, params)
    X1 = embs['src'][dico[:, 0]]
    X2 = embs['tgt'][dico[:, 1]]

    # apply the Procrustes solution
    # trainer.procrustes()
    W = train_supervision(X1, X2)

    return W

###============= Working =============##########

if params.work_mode == 'analysis':
    # evaluate_classification(params.image_data, params.src_lang, params)
    evaluate_classification(params.image_data, params.tgt_lang, params)
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
    from IPython import embed; embed()


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
    embs['src']= embs['src'].type(torch.cuda.FloatTensor)
    # test(word2ids, {'src': embs['tgt'], 'tgt': embs['tgt']})
    if params.sim_score == 'ranking':
        eval(None, embs)
    else:
        test(word2ids, embs)
    ###============= Translation =============##########
    # decorrelate
    # if params.emb_type == 'cliptext':
    #     W = scipy.stats.ortho_group.rvs(embs['src'].shape[1])
    #     W = torch.from_numpy(W).type(torch.FloatTensor).cuda()
    #     embs['src'] = embs['src'] @ W.T

   