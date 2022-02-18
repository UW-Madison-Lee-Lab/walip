
import numpy as np
import torch, os, scipy
import scipy.optimize
from scipy import stats
import torch.nn.functional as F
from evals.word_translation import get_csls_word_translation, build_dictionary, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy
from utils.text_loader import combine_files
from quad_ot.gmp import quadratic_assignment_ot
import configs



def cal_similarity(sim_score, dico, embs):
    for l in ['src', 'tgt']:
        embs[l] = F.normalize(embs[l], dim=1)

    if sim_score in ['csls', 'cosine']:
        scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], sim_score)
    elif sim_score == 'inner_prod':
        test_emb0 = embs['src'][dico[:, 0]]
        test_emb1 = embs['tgt'][dico[:, 1]]
        scores = test_emb0 @ test_emb1.T 
    elif sim_score == 'ranking':
        rank_embs = {}
        for l in ['src', 'tgt']:
            ranks = torch.topk(embs[l], 10, dim=1)[1].cpu().numpy()
            rank_embs[l] = np.zeros(embs[l].shape)
            for r in range(ranks.shape[0]):
                for k in range(10):
                    rank_embs[l][r, ranks[r, k]] = 10 - k
        scores = rank_embs['src'] @ rank_embs['tgt'].T
        scores = torch.from_numpy(scores).to(embs.device)

    return scores


def align_words(matching_method, dico, scores):
    if matching_method == 'nn':
        results = get_topk_translation_accuracy(dico, scores)
        print(results)
        s = scores.max(dim=1) # value and argmax, 
        def get_precision(s, threshold):
            correct, total, lst = 0, 0, []
            for i in range(len(scores)):
                if s[0][i] > threshold:
                    total +=1
                    lst.append([i, s[1][i].cpu().numpy()])
                    if s[1][i] == i:
                        correct += 1
            print("Precision@1 ", correct, total, correct/total)
            return lst
        
        threshold = torch.quantile(s[0], 0.5)
        print('Threshold', threshold)
        lst = get_precision(s, threshold)
        # np.save('../results/lst', np.asarray(lst))
        return np.asarray(lst)
        
    if matching_method == 'hungarian':
        cost = -scores.cpu().numpy()
        dico = dico.cpu()    
        _, col_ind = scipy.optimize.linear_sum_assignment(cost)
        acc, wrong_pairs = get_accuracy(dico, col_ind)


def load_test_dict(params, word2ids):
    test_fpath = params.txt_dir + f'{params.word_data}_{params.src_lang}_{params.tgt_lang}_{params.data_mode}.txt'
    if not os.path.isfile(test_fpath):
        combine_files(params)
    test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'], delimiter=configs.delimiters[params.word_data])
    test_dico = test_dico.cuda()
    return test_dico

###============= Supervision =============##########
def train_supervision(X1, X2):
    # X = X1.T @ X2
    # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
    M = X2.transpose(0, 1).mm(X1).cpu().numpy()
    U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
    # W = U @ VT
    W = U.dot(VT)
    return torch.Tensor(W).cuda()

def refinement(params, W, embs):
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



###============= Quadratic Assignment =============##########
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
