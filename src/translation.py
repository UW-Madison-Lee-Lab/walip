
import numpy as np
import torch, os, scipy
import scipy.optimize
from scipy import stats
import torch.nn.functional as F
from evals.word_translation import get_csls_word_translation, build_dictionary, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy, generate_path
from utils.text_loader import combine_files
import configs



def cal_similarity(sim_score, dico, embs, emb_type, row_ranking=True):
    for l in ['src', 'tgt']:
        if emb_type == 'fp':
            # t = torch.median(embs[l])
            # t = torch.quantile(embs[l], 0.9)
            t = np.quantile(embs[l].cpu().numpy(), 0.9)
            embs[l] = embs[l] * (embs[l] > t)
        embs[l] = F.normalize(embs[l], dim=1)
    scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], sim_score)
    return scores


def get_dico_dict(dico):
    d = {}
    t = dico.cpu().numpy()
    for i in range(len(t)):
        r = t[i, :]
        if r[0] in d:
            d[r[0]].append(r[1])
        else:
            d[r[0]] = [r[1]]
    return d


def get_recall(dico, scores):
    d = get_dico_dict(dico)
    s = scores.topk(10, 1, True)
    correct1, correct10 = 0, 0
    checked_ids = []
    for i in range(len(scores)):
        k = dico[i, 0].item()
        if not(k in checked_ids):
            checked_ids.append(k)
            if s[1][i, 0] in d[k]:
                correct1 += 1
                correct10 += 1
            else:
                for t in d[k]:
                    if t in s[1][i, :]:
                        correct10 += 1
                        break
    total = len(checked_ids)
    return correct1/total*100, correct10/total*100

def align_words(dico, scores, c=0.5, k=1):
    if dico is not None:
        results = get_topk_translation_accuracy(dico, scores)
        print(results)
        d = get_dico_dict(dico)

    # s = scores.max(dim=1) # value and argmax, 
    s = scores.topk(k, 1, True)
    def get_precision(s, threshold):
        correct, total, lst = 0, 0, []
        for i in range(len(scores)):
            for m in range(k):
                if s[0][i, m] > threshold:
                    total +=1
                    lst.append([i, s[1][i, m].cpu().numpy()])
                    if dico is not None and s[1][i, m] in d[dico[i, 0].item()]:
                        correct += 1
        if dico is not None:
            print("---------> Prec@1 {:.2f} {}/{}".format(correct/total*100, correct, total))
        return lst
    if c >  0:
        threshold = torch.quantile(s[0], c)
    else:
        threshold = 0
    lst = get_precision(s, threshold)
    return np.asarray(lst)



def load_test_dict(params, word2ids):
    test_fpath = generate_path('txt_pair',  {'word_data': params.word_data, 'src_lang': params.src_lang, 'tgt_lang': params.tgt_lang, 'data_mode': params.data_mode})
    if not os.path.isfile(test_fpath):
        combine_files(params.langs, params.word_data, params.data_mode)
    test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'])
    # delimiter=configs.delimiters[params.word_data])
    test_dico = test_dico.to(params.device)
    return test_dico

###============= Supervision =============##########
def train_supervision(X1, X2):
    # X = X1.T @ X2
    # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
    M = X2.transpose(0, 1).mm(X1).cpu().numpy()
    U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
    # W = U @ VT
    W = U.dot(VT)
    return torch.Tensor(W).to(X1.device)

def robust_procrustes(X, Y, c=0.1):
    n, k = X.shape
    W = train_supervision(X, Y)
    # W = torch.randn(k, k).to(X.device)
    D = torch.eye(n).to(X.device)
    X1 = X
    Y1 = Y
    for _ in range(2):
        e = ((Y1 - X1 @ W.T)**2).sum(dim=1)
        alphas = 1/(e + 0.001)
        alphas = alphas / alphas.max()
        for i in range(n):
            D[i, i] = alphas[i]**0.5
        X1 = D @ X1
        Y1 = D @ Y1
        M = Y1.T @ X1
        U, Sigma, VT = scipy.linalg.svd(M.cpu().numpy(), full_matrices=True)
        W = U @ VT
        W = torch.Tensor(W).to(X.device)
    return W
