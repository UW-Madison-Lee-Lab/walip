
import numpy as np
import torch, os, scipy
import scipy.optimize
from scipy import stats
from evals.word_translation import get_csls_word_translation, build_dictionary, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy
from utils.text_loader import combine_files
from pkg.gmp import quadratic_assignment_ot
import configs



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

def calculate_similarity(params, dico, embs):
    if params.sim_score in ['csls', 'cosine']:
        scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], params.sim_score)
    elif params.sim_score == 'inner_prod':
        test_emb0 = embs['src'][dico[:, 0]]
        test_emb1 = embs['tgt'][dico[:, 1]]
        scores = test_emb0 @ test_emb1.T 
    elif params.sim_score == 'ranking':
        def similarity(a, b):
            s = 0
            for i in range(10):
                if a[i] in b:
                    j = np.where(b == a[i])[0]
                    s += (10 - i) * (10 - j)
            return s
        ranks = {}
        for l in ['src', 'tgt']:
            ranks[l] = torch.topk(embs[l], 10, dim=1)[1].cpu().numpy()
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

def match_embeddings(params, dico, embs):
    print('\n..... Evaluating ..... ', params.word_data, params.emb_type, params.sim_score, params.matching_method)        

    ###============= Similarity Calculation =============##########
    # from IPython import embed; embed()
    scores = calculate_similarity(params, dico, embs)

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
    elif params.matching_method == 'hungarian':
        cost = -scores.cpu().numpy()
        dico = dico.cpu()    
        _, col_ind = scipy.optimize.linear_sum_assignment(cost)
        acc, wrong_pairs = get_accuracy(dico, col_ind)
    elif params.matching_method == 'quad_ot':
        quad_ot(embs)
        
def evaluate_translation(params, word2ids, embs):
    ##### Testing pairs  (subsets of dictionaries)
    test_fpath = params.txt_dir + f'{params.word_data}_{params.src_lang}_{params.tgt_lang}_{params.data_mode}.txt'
    if not os.path.isfile(test_fpath):
        combine_files(params)
    test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'], delimiter=configs.delimiters[params.word_data])
    test_dico = test_dico.cuda()
    match_embeddings(params, test_dico, embs)

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
