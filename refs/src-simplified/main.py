import os, torch, sys, scipy
import numpy as np
import torch.nn.functional as F
from utils.text_loader import get_word2id, load_vocabs, write_vocabs
from utils.helper import dict2clsattr
from translation import align_words, load_test_dict, train_supervision, cal_similarity, robust_procrustes, get_dico_dict
from models.embedding import ClipEmbedding
from evals.word_translation import get_csls_word_translation, get_topk_translation_accuracy, get_topk_accuracy, get_candidates
import argparse, json
import configs
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--save_lst", default=0, type=int)
parser.add_argument("-c", "--config", type=str, default='u')
args = parser.parse_args()
with open(f"configs/analysis.json") as f:
    model_config = json.load(f)

for s in ['reuse_embedding', 'reuse_text_embedding', 'reuse_image_embedding', 'reuse_image_data']:
    model_config["work"][s] = model_config["work"][s] == 1
args = dict2clsattr(vars(args), model_config)
args.langs = {configs.SRC: args.src_lang, configs.TGT: args.tgt_lang}
# args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.device = torch.device('cpu')

################# ================= ################
def load_200k_embeddings(langs, word_data, opts):
    word2ids, vocabs = {}, {}
    embs = {configs.FINGERPRINT: {}, configs.FASTTEXT: {}}
    data_mode = 'full'
    for key, lang in langs.items():
        vocabs[key] = load_vocabs(lang, opts.langs, word_data, data_mode)
        word2ids[key] = get_word2id(vocabs[key])
        for emb_type in [configs.FINGERPRINT, configs.FASTTEXT]:
            embObj = ClipEmbedding(emb_type, lang, data_mode, opts)
            embs[emb_type][key] = torch.Tensor(embObj.load_embedding(vocabs[key])).to(opts.device)
    return word2ids, embs, vocabs

def keep_high2(e, c=0.5):
    # max_cos = torch.topk(e, 1, dim=1)[0].sum(dim=1).cpu().numpy()
    top2 = torch.topk(e, 2, dim=1)[0].cpu().numpy()
    max_cos = top2[:, 0] - top2[:, 1]
    med = np.quantile(max_cos, c) # 0.25
    inds = np.where(max_cos > med)[0]
    return inds

def keep_high(e, c=0.5):
    max_cos = torch.topk(e, 1, dim=1)[0].sum(dim=1).cpu().numpy()
    med = np.quantile(max_cos, c) # 0.25
    inds = np.where(max_cos > med)[0]
    return inds

def cluster_noun(e, c=0.5):
    max_cos = torch.topk(e, 2, dim=1)[0].sum(dim=1).cpu().numpy()
    med = np.quantile(max_cos, c) # 0.25
    inds = np.where(max_cos > med)[0]
    return inds

def re2(top_matches, d):
    indices = top_matches.numpy()
    count = 0
    for k in range(len(indices)):
        for i in d[k]:
            if i in indices[k]:
                count+= 1
                break
    print('Recall: ', count, len(indices), count/len(indices) * 100)


# semi-supervised
def run_iter(W, fast_embs, src_test_dico, tgt_ids):
    scores = cal_similarity('csls', None, {'src': fast_embs[configs.SRC] @ W.T, 'tgt': fast_embs[configs.TGT]}, configs.FASTTEXT, False)
    # testing
    top_10_matches = scores.topk(10, 1, True)[1].numpy()
    for i in range(top_10_matches.shape[0]):
        for j in range(top_10_matches.shape[1]):
            top_10_matches[i, j] = tgt_ids[top_10_matches[i, j]]
    results = get_topk_accuracy(src_test_dico, torch.Tensor(top_10_matches[src_test_dico[:, 0], :]))
    print(results)
    return scores, top_10_matches

def align_words2(matching_method, dico, scores, c=0.5):
    k = 1
    s = scores.topk(k, 1, True)
    threshold = torch.quantile(s[0], c) #0.5 -- it
    lst = []
    for i in range(len(scores)):
        for m in range(k):
            if s[0][i, m] > threshold:
                lst.append([i, s[1][i, m].cpu().numpy()])
    return np.asarray(lst)

def robust_procrustes3(X, Y, option=1):
    n, k = X.shape
    W = torch.randn(k, k).to(X.device)
    D = torch.eye(n).to(X.device)
    X1, Y1 = X, Y
    for j in range(15):
        X_w = F.normalize(X1 @ W.T, dim=1)
        e = (1-(X_w * Y1).sum(dim=1))**2
        alphas = 1/(e + 0.001)
        alphas = alphas / alphas.max()
        for i in range(n):
            D[i, i] = alphas[i]**0.5
        M =  (D @ Y1).T @ (D @ X1)
        U, Sigma, VT = scipy.linalg.svd(M.cpu().numpy(), full_matrices=True)
        W = torch.Tensor(U @ VT).to(X.device)
    return W

###============= Working Mode =============##########
# load all embs:
word2ids, embs, vocabs = load_200k_embeddings(args.langs, args.word_data, args)

# test_dico
test_dico = load_test_dict(args, word2ids)
src_ids = []
for i in test_dico[:, 0].numpy():
    if i not in src_ids:
        src_ids.append(i)

src_ids = np.asarray(src_ids) # all 1500 ids
# src_ids = src_ids[filter_ids]
src_id_map = {}
for i in range(len(src_ids)):
    src_id_map[src_ids[i]] = i

src_test_dico = [] # [0, id]
for i in range(len(test_dico)):
    r = test_dico[i].numpy()
    if r[0] in src_ids:
        src_test_dico.append([src_id_map[r[0]], r[1]])
src_test_dico = torch.Tensor(np.asarray(src_test_dico)).type(torch.LongTensor)

# dictionary
d = get_dico_dict(src_test_dico)

### Fastttext
W_fasttext = np.load(f'../dicts/W_{args.langs[configs.SRC]}_{args.langs[configs.TGT]}.npy', allow_pickle=True)
W_fasttext = torch.Tensor(W_fasttext).to(args.device)

F0 = embs[configs.FASTTEXT][configs.SRC][src_ids]
F1 = embs[configs.FASTTEXT][configs.TGT]

# fast_scores = get_csls_word_translation(np.arange(len(src_ids)), F0@W_fasttext.T, F1, 'csls') 
fast_scores = torch.Tensor(np.load('../dicts/fast_scores_en_ru.npy', allow_pickle=True))
K_TOP = 5
fast_matches = fast_scores.topk(K_TOP, 1, True)[1]
results = get_topk_accuracy(src_test_dico, fast_matches[src_test_dico[:, 0], :])
print("Fast-text", results)

# target ids
tgt_ids = []
for i in fast_matches.flatten().numpy():
    if i not in tgt_ids:
        tgt_ids.append(i)

tgt_id_map = {}
for i in range(len(tgt_ids)):
    tgt_id_map[tgt_ids[i]] = i


### Get 7500 pairs 
candidate_lst = []
# for i in range(len(src_ids)):
#     for j in fast_matches[i].numpy():
#         candidate_lst.append([i, tgt_id_map[j]])
for i in range(len(src_ids)):
    candidate_lst.append([i, tgt_id_map[fast_matches[i, 0].item()]])
candidate_lst = np.asarray(candidate_lst)

lst_fast = []
for i in range(1500):
    if candidate_lst[i, 1] in d[i]:
        lst_fast.append(i)

# inds = keep_high2(fast_scores)
# filter_ids = []
# for i in range(len(src_ids)):
#     if i not in inds:
#         filter_ids.append(i)
# filter_ids = np.asarray(filter_ids)
# top_matches = top_matches[filter_ids, :]

# using ranking ------ from here
# align words
fp_embs = {'src':  embs['fp']['src'][src_ids, :], 'tgt': embs['fp']['tgt'][tgt_ids, :]}
# noun_ind_src= np.arange(len(fp_embs['src'])) #cluster_noun(fp_embs['src'])
# noun_ind_tgt = np.arange(len(fp_embs['tgt']))  #cluster_noun(fp_embs['tgt'])
fp_scores = cal_similarity('csls', np.arange(src_ids.shape[0]), {'src': fp_embs['src'], 'tgt': embs['fp']['tgt']}, 'fp', False) 
# all_fp_scores = cal_similarity('csls', np.arange(len(noun_ind_src)), {'src': fp_embs['src'][noun_ind_src, :], 'tgt': fp_embs['tgt'][noun_ind_tgt, :]}, 'fp', False) 

# fp_scores = torch.Tensor(np.load('../dicts/fp_scores_en_ru.npy', allow_pickle=True))
# noun_inds = align_words('nn', None, fp_scores)
# for i in range(len(noun_inds)):
#     r = noun_inds[i]
#     noun_inds[i, 0] = noun_ind_src[r[0]]
#     noun_inds[i, 1] = noun_ind_tgt[r[1]]
# fingerprint
# fp_matches = []
# for i in range(len(src_ids)):
#     y_fp_id = fast_matches[i].numpy()
#     y_ids = []
#     for k in y_fp_id:
#         y_ids.append(tgt_id_map[k])
#     topk = fp_scores[i, y_ids].topk(K_TOP)[1].numpy()
#     real_topk = []
#     for j in topk:
#         real_topk.append(y_fp_id[j])
#     fp_matches.append(real_topk)

# fp_matches2 = torch.Tensor(np.asarray(fp_matches)).type(torch.LongTensor)
# results = get_topk_accuracy(src_test_dico, fp_matches[src_test_dico[:, 0], :] )
# print(results)
fp_candidate_scores = []
for i in range(len(candidate_lst)):
    r =  candidate_lst[i]
    fp_candidate_scores.append([fp_scores[r[0], r[1]]])

inds = keep_high(torch.Tensor(np.asarray(fp_candidate_scores)))
# semi-supervised
fast_embs = {configs.SRC: F0[candidate_lst[:, 0]], configs.TGT: F1[tgt_ids][candidate_lst[:, 1]]}
# W_fp = robust_procrustes(fast_embs['src'][inds], fast_embs['tgt'][inds])
# W_fp = train_supervision(fast_embs['src'], fast_embs['tgt'])

# candidate_lst: 1500
# lst_fp = [] 
# W_fp = train_supervision(fast_embs['src'][lst], fast_embs['tgt'][lst])
# scores, top_10_matches = run_iter(W_fp, fast_embs2, src_test_dico, tgt_ids)

x_fp = embs['fp']['src'][src_ids]
y_fp = embs['fp']['src'][fast_matches[:, 0].numpy()]

nrows, ncols = x_fp.shape

def cal_ranking_sim(emb, N_RANK):
    rank_emb = torch.zeros(emb.shape).to(emb.device)
    nrows, ncols = emb.shape
    ranks = torch.topk(emb, N_RANK, dim=1)[1]# nrows x k
    for r in range(nrows):
        for k in range(N_RANK):
            ind = ranks[r, k]
            if emb[r, ind] > 0:
                rank_emb[r, ind] = N_RANK - k
    # ranks = torch.topk(emb, N_RANK, dim=0)[1]
    # for c in range(ncols):
    #     for k in range(N_RANK):
    #         ind = ranks[k, c]
    #         rank_emb[ind, c] = N_RANK - k
    # rank_emb = F.normalize(rank_emb, dim=1)
    return rank_emb

from IPython import embed; embed()


N_RANK = 10
x_rank = cal_ranking_sim(x_fp, N_RANK)
y_rank = cal_ranking_sim(y_fp, N_RANK)

x_mean = x_fp.mean(dim=0)
x_mean = x_mean.unsqueeze(0).expand(x_fp.shape)
x_std = x_fp.std(dim=0)
x_std = x_std.unsqueeze(0).expand(x_fp.shape)

y_mean = y_fp.mean(dim=0)
y_mean = y_mean.unsqueeze(0).expand(y_fp.shape)
y_std = y_fp.std(dim=0)
y_std = y_std.unsqueeze(0).expand(y_fp.shape)

x_rank = (x_fp - x_mean)/x_std
y_rank = (y_fp - y_mean)/y_std

t_x = torch.quantile(x_fp, 0.9, dim=1)
t_x = t_x.unsqueeze(1).expand(x_fp.shape)
x_rank = x_fp * (x_fp > t_x)
x_rank = F.normalize(x_rank, dim=1)

t_y = torch.quantile(y_fp, 0.9, dim=1)
t_y = t_y.unsqueeze(1).expand(y_fp.shape)
y_rank = y_fp * (y_fp > t_y)
y_rank = F.normalize(y_rank, dim=1)


s = (x_rank * y_rank).sum(dim=1)
inds = np.where(s.numpy() < torch.quantile(s, 0.1).item())[0]
count = 0
for i in inds:
    if i in lst_fast:
        count += 1
# scores = rank_embs['src'] @ rank_embs['tgt'].T
scores = get_csls_word_translation(dico, rank_embs['src'], rank_embs['tgt'], 'csls')

#supervised
X = F0[src_test_dico[:, 0]]
Y = F1[src_test_dico[:, 1]]
W_true = train_supervision(X, Y)


W = W_fp
fast_embs2 = {configs.SRC: F0, configs.TGT: F1[tgt_ids]}
for t in range(10):  # refinement
    scores, top_10_matches = run_iter(W, fast_embs2, src_test_dico, tgt_ids)
    inds = align_words2('nn', None, scores, c=0.5/(t+1))
    W = robust_procrustes3(fast_embs[configs.SRC][inds[:, 0]], fast_embs[configs.TGT][inds[:, 1]], option=2)

from IPython import embed; embed()
score2 = scores.topk(1, 1, True)[0].flatten().numpy()
score1 = fast_scores.topk(1, 1, True)[0].flatten().numpy()
fp_lst = top_10_matches[:, 0].numpy()
fast_lst = fast_matches[:, 0]

# s_lst = s_scores.topk(1, 1, True)[1].flatten().numpy()
# for i in range(len(s_lst)):
#     s_lst[i] = tgt_ids[s_lst[i]]

# %autoindent, %paste , %cpaste
lst_same = []
lst_fp = []
lst_fast = []
for i in range(1500):
    if fast_lst[i] == fp_lst[i]:
        lst_same.append(i)
    if fp_lst[i] in d[i]:
        lst_fp.append(i)
    if fast_lst[i] in d[i]:
        lst_fast.append(i)

##### Testing pairs  (subsets of dictionaries)
count = 0
for i in range(len(inds)):
    k = inds[i]
    if k in lst_fast:
        count += 1

print(count)

src_emb = embs[configs.FASTTEXT][configs.SRC]
tgt_emb = embs[configs.FASTTEXT][configs.TGT]
emb1 = F.normalize(src_emb @ W_fasttext.T, dim=1)
emb2 = F.normalize(embs[configs.FASTTEXT][configs.TGT], dim=1)
val_dico = get_candidates(emb1, emb2, params)
W_val = robust_procrustes3(src_emb[val_dico[:, 0]], tgt_emb[val_dico[:, 1]])

val_parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
val_parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
val_parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
val_parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
val_parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
val_parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
val_parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
val_parser.add_argument("-c", "--config", type=str, default='u')
params = val_parser.parse_args()


X_val = embs['fp']['src'][val_dico[:, 0]]
t = torch.quantile(X_val, 0.5)
X_val = X_val * (X_val > t)
X_val = F.normalize(X_val, dim=1)

Y_val = embs['fp']['tgt'][val_dico[:, 1]]
t = torch.quantile(Y_val, 0.5)
Y_val = Y_val * (Y_val > t)
Y_val = F.normalize(Y_val, dim=1)
T = (X_val * Y_val).sum(dim=1)
c = torch.quantile(T, 0.5)
mask = (T>c).unsqueeze(1).expand_as(val_dico).clone()
all_scores = val_dico.masked_select(mask).view(-1, 2)
W_val = robust_procrustes3(src_emb[all_scores[:, 0]], tgt_emb[all_scores[:, 1]])