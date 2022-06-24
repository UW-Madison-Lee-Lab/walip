import os, torch, sys
import numpy as np
import argparse, json
import torch.nn.functional as F

from models.embedding import ClipEmbedding
from translation import align_words, load_test_dict, train_supervision, cal_similarity, robust_procrustes, get_recall
from evals.word_translation import read_txt_embeddings
from utils.helper import dict2clsattr, generate_path
from utils.text_loader import get_word2id, load_vocabs, load_vocabs_from_pairs, write_vocabs

import configs
os.environ['TOKENIZERS_PARALLELISM'] = "false"

######  MAIN ##############
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("-s", "--src_lang", default='en', type=str, required=True)
parser.add_argument("-t", "--tgt_lang", default='it', type=str, required=True)
parser.add_argument("-w", "--work_mode", type=str, default='c', help="Working mode [b: baseline, c: cuwt, s: supervision") 
parser.add_argument("-b", "--baseline", type=str, default='muse', help="Working mode [muse, muve, globe, nn") 
parser.add_argument("-p", "--cuwt_phase", type=str, default='t', help="Working phase [c: cluster, u: unsupervised, t: translation") 
parser.add_argument("-m", "--translation", type=str, default='s', help="Working phase [s: semi, z: zero-shot, t: transition, z:random")
parser.add_argument("--proc", type=str, default='robust', help='[robust, normal]')
parser.add_argument("-d", "--debug", default=0, type=int)
parser.add_argument("-g", "--gpu_id", default=1, type=int)

working_modes = {'b': 'baseline', 'c': 'cuwt', 's': 'supervision'}
phases = {"character_frequency":"character_frequency","substring_matching": "substring_matching", 'c': 'cluster', 'u': 'unsupervised', 't': 'translation'}

######  Load configurations ##############
args = parser.parse_args()

# Load general config
with open(f"configs/settings.json") as f:
    general_configs = json.load(f)
# Load working config
with open(f"configs/{working_modes[args.work_mode]}.json") as f:
    working_configs = json.load(f)
if 'c' == args.work_mode:
    print(working_configs)
    working_configs = working_configs[phases[args.cuwt_phase]]

model_configs = {**general_configs, **working_configs}
args = dict2clsattr(vars(args), model_configs)
# Load langs config
with open(f"configs/langs.json") as f:
    langs_configs = json.load(f)
args.langs = {'src': args.src_lang, 'tgt': args.tgt_lang}
args.lang_configs = {args.src_lang: langs_configs[args.src_lang], args.tgt_lang: langs_configs[args.tgt_lang]}
args.large_model = langs_configs[args.tgt_lang]["large_model"]

if args.gpu_id == -1 or not(torch.cuda.is_available()):
    args.device = torch.device('cpu')
else:
    args.device = torch.device(f'cuda:{args.gpu_id}')

################# ======= Functions ========== ################
def get_mutual_nn(src_embs, tgt_embs, dist_fn, k = 5):
    """
    """
    src_indices = np.arange(len(src_embs))
    tgt_indices = np.arange(len(tgt_embs))
    pairs = []
    for src_index in range(len(src_embs)):
        topktgt = sorted(tgt_indices, key = lambda j: dist_fn(src_embs[src_index], tgt_embs[j]))[:k]
        for tgt_index in topktgt:
            topksrc = sorted(src_indices, key = lambda j: dist_fn(tgt_embs[tgt_index], src_embs[j]))[:k]
            for idx in topksrc:
                if idx == src_index:
                    pairs.append((src_index, tgt_index))
    pair_indices = np.array(pairs)
    filtered_src_embs = src_embs[pair_indices[:,0]]
    filtered_tgt_embs = tgt_embs[pair_indices[:,1]]
    return pair_indices, filtered_src_embs, filtered_tgt_embs

def load_word2ids(langs, word_data, data_mode):
    word2ids = {}
    for key, lang in langs.items():
        vocab = load_vocabs(lang, langs, word_data, data_mode)
        word2ids[key] = get_word2id(vocab)
    return word2ids

def load_embedding(lang, emb_type, word_data, data_mode, opts):
    vocab = load_vocabs(lang, opts.langs, word_data, data_mode)
    embObj = ClipEmbedding(emb_type, lang, data_mode, opts)
    embs = torch.from_numpy(embObj.load_embedding(vocab)).to(opts.device)
    word2ids = get_word2id(vocab)
    return word2ids, embs

def load_two_embeddings(langs, emb_type, word_data, data_mode, opts):
    word2ids, embs = {}, {}
    for key, lang in langs.items():
       word2ids[key], embs[key] = load_embedding(lang, emb_type, word_data, data_mode, opts)
    return word2ids, embs

###### Clustering Nouns 

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

def check_correct_pair(indices, dico):
    d = get_dico_dict(dico)
    count = 0
    correct_pairs = []
    for i in range(len(indices['src'])):
        x, y = indices['src'][i], indices['tgt'][i]
        if x in d and y in d[x]:
            count += 1
            correct_pairs.append([x, y])
    print(' === Noun Acc: {:.2f}% =='.format(100 * count/len(indices['src'])))
    return d, correct_pairs

def convert_index(test_dico, args):
    src_ids = []
    for i in test_dico[:, 0].cpu().numpy():
        if i not in src_ids:
            src_ids.append(i)

    src_id_map = {}
    for i in range(len(src_ids)):
        src_id_map[src_ids[i]] = i

    src_test_dico = []
    for i in range(len(test_dico)):
        r = test_dico[i].cpu().numpy()
        src_test_dico.append([src_id_map[r[0]], r[1]])
    src_test_dico = torch.Tensor(np.asarray(src_test_dico)).type(torch.LongTensor).to(args.device)
    return src_test_dico, src_ids, src_id_map

def get_indices_from_nouns(inds, word2ids, src_id_map, args):
    nouns = {}
    for l in ['src', 'tgt']:
        c = 0 if l == 'src' else 1
        vocab = load_vocabs(args.langs[l], args.langs, args.word_data, args.data_mode)
        nouns[l] = [vocab[i] for i in inds[:, c]]


    indices = {'src': [], 'tgt': []}
    for i in range(len(nouns['src'])):
        w1, w2 = nouns['src'][i], nouns['tgt'][i]
        if w1 in word2ids['src'] and w2 in word2ids['tgt'] and word2ids['src'][w1] in src_id_map:
            indices['src'].append(src_id_map[word2ids['src'][w1]])
            indices['tgt'].append(word2ids['tgt'][w2])
    return indices

def num_common_chars_v2(s1, s2):
    intersection = set(s1) & set(s2)

def num_common_chars(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return set1 & set2

def lcs(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range (1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]
###============= Working Mode =============##########
if args.work_mode == 'c': # CUWT
    ###============= Clustering =============##########
    if args.cuwt_phase == 'c': # cluster
        _, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
        for l in ['src', 'tgt']:
            vocab = load_vocabs(args.langs[l], args.langs, args.word_data, args.data_mode)
            fp = embs[l] # n x m
            max_cos = torch.topk(fp, 2, dim=1)[0].sum(dim=1).cpu().numpy()
            med = np.quantile(max_cos, 0.5)
            ind = np.where(max_cos > med)[0]
            nouns, new_ind = [], []
            for i in ind:
            
                if vocab[i] not in nouns:
                    new_ind.append(i)
                    nouns.append(vocab[i])

            # output noun vocabs
            noun_embs = fp[new_ind]
            noun_data = args.word_data + '_noun'
            write_vocabs(nouns, args.langs[l], args.langs, noun_data, args.data_mode)
            # save fp
            emb_path = generate_path('emb_fp', {'lang': args.langs[l], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang, 'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test', 'selected': args.using_filtered_images, 'num_images': args.num_images})
            np.save(emb_path, noun_embs.cpu().numpy())
            print('Save emb')
        sys.exit("DONE!!!!") 
    elif args.cuwt_phase == "substring_matching":
        threshold = 0.75
        noun_data = args.word_data + "_noun"
        lst_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'
        _, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
        src_vocab = load_vocabs(args.langs['src'], args.langs, args.word_data, args.data_mode)
        tgt_vocab = load_vocabs(args.langs['tgt'], args.langs, args.word_data, args.data_mode)

        pairs = []
        indices = []
        src_noun = []
        tgt_noun = []
        src_indices = []
        tgt_indices = []
        for i, src_word in enumerate(src_vocab):
            matches = [(src_word, tgt, i, j) for (j, tgt) in enumerate(tgt_vocab) if len(lcs(src_word, tgt)) >= threshold * max(len(src_word), len(tgt))]
            for match in matches:
                indices.append([match[2], match[3]])
                if match[0] not in src_noun:
                    src_indices.append(match[2])
                    src_noun.append(match[0])
                if match[1] not in tgt_noun:
                    tgt_indices.append(match[3])
                    tgt_noun.append(match[1])
        np.save(lst_path, np.array(indices))
        print(f'Found {len(indices)} substring matches')
        write_vocabs(src_vocab, args.langs['src'], args.langs, noun_data, args.data_mode)
        write_vocabs(tgt_vocab, args.langs['tgt'], args.langs, noun_data, args.data_mode)
        emb_path = generate_path('emb_fp', {'lang': args.langs['src'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang, 'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test', 'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embs['src'].cpu().numpy())
        emb_path = generate_path('emb_fp', {'lang': args.langs['tgt'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang, 'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test', 'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embs['tgt'].cpu().numpy())
        sys.exit("Finished substring matching")


    elif args.cuwt_phase == 'character_frequency':
        _, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
        threshold = 0.75
        noun_data = args.word_data + "_noun"
        lst_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'
        src_vocab_orig = load_vocabs(args.langs['src'], args.langs, args.word_data, args.data_mode)
        tgt_vocab_orig = load_vocabs(args.langs['tgt'], args.langs, args.word_data, args.data_mode)
        src_vocab = load_vocabs(args.langs['src'], args.langs, args.word_data, args.data_mode)
        tgt_vocab = load_vocabs(args.langs['tgt'], args.langs, args.word_data, args.data_mode)

        src_counts = {}
        tgt_counts = {}
        character_mapping = {}
        for word in src_vocab_orig:
            characters = [char for char in word]
            for character in characters:
                if character not in src_counts:
                    src_counts[character] = 0
                src_counts[character] += 1

        for word in tgt_vocab_orig:
            characters = [char for char in word]
            for character in characters:
                if character not in tgt_counts:
                    tgt_counts[character] = 0
                tgt_counts[character] += 1
        src_sorted = sorted(src_counts, key = lambda x: src_counts[x], reverse = True)
        tgt_sorted = sorted(tgt_counts, key = lambda x: tgt_counts[x], reverse = True)

        for i in range(min(len(src_sorted), len(tgt_sorted))):
            character_mapping[tgt_sorted[i]] = src_sorted[i]
        tgt_vocab = []
        for word in tgt_vocab_orig:
            translated_word = ""
            for character in word:
                if character in character_mapping:
                    translated_word += character_mapping[character]
                else:
                    translated_word += character
            tgt_vocab.append(translated_word)

        indices = [] 
        src_noun = []
        tgt_noun = []
        src_indices = []
        tgt_indices = []
        for i, src_word in enumerate(src_vocab):
            matches = [(src_word, tgt, tgt_vocab_orig[j], i, j) for (j, tgt) in enumerate(tgt_vocab) if len(lcs(src_word, tgt)) >= threshold * max(len(src_word), len(tgt))]
            for match in matches:
                print(match)
                indices.append([match[3], match[4]])
                if match[0] not in src_noun:
                    src_indices.append(match[2])
                    src_noun.append(match[0])
                if match[1] not in tgt_noun:
                    tgt_indices.append(match[3])
                    tgt_noun.append(match[1])
        


        np.save(lst_path, np.array(indices))
        print(f"Found {len(indices)} character match pairs")
        write_vocabs(src_vocab_orig, args.langs['src'], args.langs, noun_data, args.data_mode)
        write_vocabs(tgt_vocab_orig, args.langs['tgt'], args.langs, noun_data, args.data_mode)
        emb_path = generate_path('emb_fp', {'lang': args.langs['src'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang, 'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test', 'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embs['src'].cpu().numpy())
        emb_path = generate_path('emb_fp', {'lang': args.langs['tgt'], 'src_lang': args.src_lang, 'tgt_lang': args.tgt_lang, 'word_data': noun_data, 'image_data': args.image_data, 'data_mode': 'test', 'selected': args.using_filtered_images, 'num_images': args.num_images})
        np.save(emb_path, embs['tgt'].cpu().numpy())
        sys.exit("Finished character freq")

    else: # semi, zero-shot, transition
        ###============= Unsupervised =============##########
        orig_data = args.word_data
        noun_data = args.word_data + '_noun'
        lst_path = f'../results/indices/indices_{args.src_lang}_{args.tgt_lang}_{noun_data}_{args.large_model}.npy'
        if args.cuwt_phase == 'u': # unsupervised
            args.word_data = noun_data
            word2ids, embs = load_two_embeddings(args.langs, args.emb_type, noun_data, args.data_mode, args)
            #id2words
            scores = cal_similarity(args.sim_score, None, embs, args.emb_type) 
            inds = align_words(None, scores, 0.7, k=1) # 0.5, k=5
            # testing
            args.word_data = orig_data
            word2ids_wiki = load_word2ids(args.langs, args.word_data, args.data_mode)
            test_dico = load_test_dict(args, word2ids_wiki)
            # Noun-data
            args.word_data = noun_data
            src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
            indices = get_indices_from_nouns(inds, word2ids_wiki, src_id_map, args)

            check_correct_pair(indices, src_test_dico)
            if not(args.debug):
                np.save(lst_path, inds)
        else:
            ###============= Semi Translation =============##########
            mapping_dir = '../results/mapping/ours'
            if args.emb_type == 'mixed':
                word2ids, embs = {}, {}
                word2ids['src'], embs['src'] = load_embedding(args.langs['src'],configs.HTW, args.word_data, args.data_mode, args)
                word2ids['tgt'], embs['tgt'] = load_embedding(args.langs['tgt'], configs.FASTTEXT, args.word_data, args.data_mode, args)
            else:
                word2ids, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
            # dico
            test_dico = load_test_dict(args, word2ids)
            src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
            for l in ['src', 'tgt']:
                embs[l] = embs[l].type(torch.FloatTensor).to(args.device)
            
            embs['src'] = embs['src'][src_ids, :] # 1500
            if args.translation == "s": # semi
                if args.emb_type == "globe":
                    import scipy
                    X0 = embs['src'][test_dico[:, 0]]#[ind0, :]
                    X1 = embs['tgt'][test_dico[:, 1]]
                    M = X1.transpose(0, 1).mm(X0).cpu().numpy()
                    U, Sigma, VT = scipy.linalg.svd(M, full_matrices=True)
                    W = U.dot(VT)
                else:
                    args.word_data = noun_data
                    inds = np.load(lst_path, allow_pickle=True)
                    indices = get_indices_from_nouns(inds, word2ids, src_id_map, args)
                    check_correct_pair(indices, src_test_dico)
                    X0 = embs['src'][indices['src'], :]
                    X1 = embs['tgt'][indices['tgt'], :]
                if args.proc == 'robust':
                    W = robust_procrustes(X0, X1)
                else:
                    W = train_supervision(X0, X1)
            elif args.translation == "r": # random
                N = len(src_id_map) // 2
                indices = {'src': np.random.randint(0, len(word2ids['src'].values()), N), 'tgt': np.random.randint(0, len(word2ids['tgt'].values()), N)}
                check_correct_pair(indices, src_test_dico)
                X0 = embs['src'][indices['src'], :]
                X1 = embs['tgt'][indices['tgt'], :]
                W = robust_procrustes(X0, X1)
            else:
                if args.translation == 'z': # zero-shot
                    pretrain_w_path = f'{mapping_dir}/W_{args.emb_type}_{args.pretrained}.npy'
                    W = np.load(pretrain_w_path, allow_pickle=True)
                else: # transition
                    w1_path = f'{mapping_dir}/W_{args.emb_type}_{args.src_lang}_{args.pretrained}.npy'
                    w2_path = f'{mapping_dir}/W_{args.emb_type}_{args.pretrained}_{args.tgt_lang}.npy'
                    W1 = np.load(w1_path, allow_pickle=True)
                    W2 = np.load(w2_path, allow_pickle=True)
                    W = W2.dot(W1)
                W = torch.from_numpy(W).to(args.device)

            def run(embs, W, dico=None, k=None, c=None):
                scores = cal_similarity(args.sim_score, dico, {'src': embs['src'] @ W.T, 'tgt': embs['tgt']}, args.emb_type) 
                if dico is not None:
                    return get_recall(dico, scores)
                else:
                    return align_words(None, scores, c, k)

            def estimate_validation(X, Y):
                return ((Y - X)**2).sum(dim=1).mean()
            
            def adapt_hyperparams(j):
                if j < 10:
                    k = 10
                    c = 0.5
                elif j < 20:
                    k = 5
                    c = 0.5
                elif k < 30:
                    k = 3
                    c = 0.3
                else:
                    k = 1
                    c = 0.1
                return k, c

            if args.proc == 'normal':
                best_W = W
                ind0 = run(embs, W, None, 1, 0)
                r1, r10 = run(embs, W, src_test_dico)
                X0, X1 = embs['src'][ind0[:, 0], :], embs['tgt'][ind0[:, 1], :]
                best_error = estimate_validation(X0@W.T, X1)
                for j in range(40):
                    inds = run(embs, W, None, 1, 0)
                    X0, X1 = embs['src'][inds[:, 0], :], embs['tgt'][inds[:, 1], :]
                    W = train_supervision(X0, X1)
                    r1, r10 = run(embs, W, src_test_dico)
                    # evaluate 
                    ind0 = run(embs, W, None, 1, 0)
                    X0, X1 = embs['src'][ind0[:, 0], :], embs['tgt'][ind0[:, 1], :]
                    error = estimate_validation(X0@W.T, X1)
                    # accuracy 
                    print("== Recall@1-10:  {:.2f}  {:.2f}, error {:.2f} == \n".format(r1, r10, error))
                    if error <= best_error:
                        best_error = error 
                        best_W = W
                        best_recall = r1
                print('Best recall {:.2f}, error {:.2f}'.format(best_recall, best_error))
            elif args.proc == 'globe':
                best_W = W
                ind0 = run(embs, W, None, 1, 0)
                r1, r10 = run(embs, W, test_dico)
                X0, X1 = embs['src'][ind0[:, 0], :], embs['tgt'][ind0[:, 1], :]
                error = estimate_validation(X0@W.T, X1)
                print("== Recall@1-10:  {:.2f}  {:.2f}, error {:.2f} == \n".format(r1, r10, error))
                # accuracy
            else:
                best_W = W
                ind0 = run(embs, W, None, 1, 0)
                r1, r10 = run(embs, W, src_test_dico)
                X0, X1 = embs['src'][ind0[:, 0], :], embs['tgt'][ind0[:, 1], :]
                error = estimate_validation(X0@W.T, X1)
                # accuracy
                print("== Recall@1-10:  {:.2f}  {:.2f}, error {:.2f} == \n".format(r1, r10, error))
                best_recall = r1
                best_error = error
                for j in range(40):  # refinement
                    k, c = adapt_hyperparams(j)
                    print('Iter', j, 'k', k, 'c', c)
                    # identify the highly similar pairs
                    inds = run(embs, W, None, k , c)
                    X0, X1 = embs['src'][inds[:, 0], :], embs['tgt'][inds[:, 1], :]
                    W = robust_procrustes(X0, X1, 0.1)
                    r1, r10 = run(embs, W, src_test_dico)
                    # evaluate 
                    ind0 = run(embs, W, None, 1, 0)
                    X0, X1 = embs['src'][ind0[:, 0], :], embs['tgt'][ind0[:, 1], :]
                    error = estimate_validation(X0@W.T, X1)
                    # accuracy 
                    print("== Recall@1-10:  {:.2f}  {:.2f}, error {:.2f} == \n".format(r1, r10, error))
                    if error <= best_error:
                        best_error = error 
                        best_W = W
                        best_recall = r1
                print('Best recall {:.2f}, error {:.2f}'.format(best_recall, best_error))
                
            if not(args.debug):
                np.save(f'{mapping_dir}/W_{args.emb_type}_{args.src_lang}_{args.tgt_lang}', W.cpu().numpy())

elif args.work_mode == 'b' and args.translation == 'nn': 
    args.emb_type = 'fp'
    word2ids, embs = load_two_embeddings(args.langs, args.emb_type, args.word_data, args.data_mode, args)
    test_dico = load_test_dict(args, word2ids)
    src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
    # eng: word to image: top1 index -- NN
    # ita: image to word: top1 index 
    
    src_w2i = torch.argmax(embs['src'], dim=1).cpu().numpy()
    tgt_i2w = torch.argmax(embs['tgt'], dim=0).cpu().numpy()
    pairs = []
    for i in range(len(src_w2i)):
        img_ind = src_w2i[i]
        word_ind = tgt_i2w[img_ind]
        pairs.append([i, word_ind])
    # test-recall
    d = get_dico_dict(src_test_dico)
    correct1 = 0
    for i in range(len(pairs)):
        if pairs[i][1] in d[i]:
            correct1 += 1
    print('Acc: ', (correct1/len(pairs) * 100))
else:
     ###============= Supervision =============##########
    if args.work_mode == 's': #supervision
        root = f"../dicts/embeddings/fasttext/wiki/fasttext_wiki_{args.langs['src']}_{args.langs['tgt']}"
        X = {}
        if not os.path.isfile(root + f"_{args.langs['src']}_val.npy"):
            # ### Get MUSE vocabs
            print('Loading MUSE pairs')
            vocabs = load_vocabs_from_pairs(args.langs, args.word_data, 'val', duplicate=True)
            word2ids, full_embs = {}, {}
            for key, lang in args.langs.items():
                emb_pth = f'../datasets/wiki/wiki.{lang}.vec'
                print('Loading ', emb_pth)
                _, word2ids[key], full_embs[key] = read_txt_embeddings(emb_pth)
            # ### Get HTW data
            embs = {}
            for key, lang in args.langs.items():
                inds = [word2ids[key][w] for w in vocabs[key]]
                embs[key] = full_embs[key][np.asarray(inds)]
                np.save(root + f"_{args.langs[key]}_val", embs[key])
                print('Done', lang, len(inds))
                X[key] = torch.from_numpy(embs[key]).type(torch.FloatTensor).to(args.device)
        else:
            # training
            word2ids, embs = load_two_embeddings(args.langs, args.emb_type, 'wiki', 'val', args)
            test_dico = load_test_dict(args, word2ids)
            for l in ['src', 'tgt']:
                embs[l] = embs[l].type(torch.FloatTensor).to(args.device)
            X['src'] = embs['src'][test_dico[:, 0]]
            X['tgt'] = embs['tgt'][test_dico[:, 1]]
        W = train_supervision(X['src'], X['tgt'])

    else: # baseline
        ###============= Baselines =============##########
        w_name = f'../results/mapping/baselines/W_{args.src_lang}_{args.tgt_lang}_{args.method}'
        if os.path.isfile(w_path := w_name + '.pth'):
            W = torch.FloatTensor(torch.load(w_path)).to(args.device)
        else:
            W = np.load(w_name + '.npy', allow_pickle=True)
            W = torch.from_numpy(W).to(args.device)
    
    # testing
    word2ids, embs = load_two_embeddings(args.langs, args.emb_type,  args.word_data, args.data_mode, args)
    test_dico = load_test_dict(args, word2ids)
    src_test_dico, src_ids, src_id_map = convert_index(test_dico, args)
    for l in ['src', 'tgt']:
        embs[l] = embs[l].type(torch.FloatTensor).to(args.device)

    scores = cal_similarity(args.sim_score, test_dico, {'src': embs['src'] @ W.T, 'tgt': embs['tgt']}, args.emb_type) 
    print(get_recall(src_test_dico, scores))
