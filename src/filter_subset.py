import os, torch, sys
import numpy as np
import torch.nn.functional as F
from scipy import stats
from tclip.clip_ops import zero_shot_classification
from utils.text_loader import load_vocabs, load_vocabs_from_pairs, write_vocabs, combine_files, get_word2id
from utils.helper import dict2clsattr, check_noun, generate_path
from translation import align_words, load_test_dict, train_supervision, cal_similarity, robust_procrustes
from evals.word_translation import load_dictionary, get_csls_word_translation, get_topk_translation_accuracy, read_txt_embeddings
from models.embedding import ClipEmbedding
from models.ops import load_models
import argparse, json
import configs
os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--save_lst", default=0, type=int)
parser.add_argument("-c", "--config", type=str)
args = parser.parse_args()
with open("configs/analysis.json") as f:
    model_config = json.load(f)
for s in ['reuse_embedding', 'reuse_text_embedding', 'reuse_image_embedding', 'reuse_image_data']:
    model_config["work"][s] = model_config["work"][s] == 1
args = dict2clsattr(vars(args), model_config)
args.langs = {'src': args.src_lang, 'tgt': args.tgt_lang}
args.device = torch.device('cpu') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################# ================= ################
def get_topk_accuracy(dico, top_matches):
    results = []
    # top_matches = scores.topk(10, 1, True)[1]
    # len(dico[:, 0]) x 10
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        # logger.info("%i source words - %s - Precision at k = %i: %f" %
                    # (len(matching), method, k, precision_at_k))
        results.append('prec_@{}: {:.2f}'.format(k, precision_at_k))
    return results


def recall(indices, test_indices):
    indices = list(set(indices.numpy()))
    count = 0
    for i in indices:
        if i in test_indices:
            count += 1
    
    print('Recall: ', count, count/len(test_indices) * 100)


def re2(top_matches, d):
indices = top_matches.numpy()
count = 0
    for k in range(len(indices)):
        for i in d[k]:
            if i in indices[k]:
                count+= 1
                break

    print('Recall: ', count, len(indices) count/len(indices) * 100)

def fp_quantile(em, c=0.9):
    t = torch.quantile(em, c, dim=1)
    t = t.unsqueeze(1).expand(em.shape)
    # t = np.quantile(embs[l].cpu().numpy(), 0.9, axis=1)
    out = em * (em > t)
    return F.normalize(out, dim=1)

def prepare_embeddings(emb_type, lang, langs, word_data, data_mode, opts, multi_embs=False):
    vocab = load_vocabs(lang, langs, word_data, data_mode)
    embObj = ClipEmbedding(emb_type, lang, data_mode, opts)
    embs = torch.from_numpy(embObj.load_embedding(vocab)).to(opts.device)
    word2ids = get_word2id(vocab)
    return word2ids, embs, vocab


# fasttext and vocabs
W = np.load('../dicts/W_en_ru.npy', allow_pickle=True)
W = torch.Tensor(W)
fast_embeddings = {}
word2ids, embs, vocabs = {}, {}, {}
mode = {'src': 'test', 'tgt': 'full'}
for l in ['src', 'tgt']:
    # emb_path = f'../muse/data/wiki.{args.langs[l]}.vec'
    # fast_id2word[l], fast_word2id[l], fast_embeddings[l] = read_txt_embeddings(emb_path, emb_dim=300, full_vocab=False)
    test_fpath = generate_path('emb_fasttext', {'word_data': args.word_data, 'lang': args.langs[l], 'src_lang': args.langs['src'], 'tgt_lang': args.langs['tgt'], 'data_mode': 'full'})
    fast_embeddings[l] = np.load(test_fpath, allow_pickle=True)
    fast_embeddings[l] = torch.Tensor(fast_embeddings[l])
    # v = [fast_id2word[l][i] for i in range(200000)]
    # write_vocabs(v, args.langs[l], args.langs, args.word_data, 'full')
    word2ids[l], embs[l], vocabs[l] = prepare_embeddings('fp', args.langs[l], args.langs, args.word_data, 'full', args)

def cluster_noun(fp):
    max_cos = torch.topk(fp, 2, dim=1)[0].sum(dim=1).numpy()
    med = np.quantile(max_cos, 0.5) # 0.25
    ind = np.where(max_cos > med)[0]
    return ind

test_fpath = generate_path('txt_pair', {'word_data': args.word_data, 'src_lang': args.langs['src'], 'tgt_lang': args.langs['tgt'], 'data_mode': 'test'})
test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'], delimiter=configs.delimiters[args.word_data])


# map test_dico into [0,.]
id_map = {}
src_id = []
id_counter = 0
new_dico = []
positions = []
for i in range(test_dico.shape[0]):
    r = test_dico[i].numpy()
    if r[0] not in id_map:
        id_map[r[0]] = id_counter
        src_id.append(r[0])
        id_counter += 1
        positions.append(i)
    new_dico.append([id_map[r[0]], r[1]])

new_dico = torch.Tensor(np.asarray(new_dico)).type(torch.LongTensor)

src_fp = embs['src'][src_id, :]
src_fast = fast_embeddings['src'][src_id, :]
tgt_fp = embs['tgt']
tgt_fast = fast_embeddings['tgt']

X0 = F.normalize(fast_embeddings['src']@W.T, dim=1)
X1 = F.normalize(tgt_fast, dim=1)

c = 0.1
P0 = fp_quantile(embs['src'], c)
P1 = fp_quantile(embs['tgt'], c)

all_fast_scores = get_csls_word_translation(test_dico, X0, X1, 'csls') 
all_fp_scores = get_csls_word_translation(test_dico, P0, P1, 'csls') 

fast_scores = all_fast_scores[positions, :]
top_matches = fast_scores.topk(10, 1, True)[1]
top_matches = top_matches[new_dico[:, 0], :]
results = get_topk_accuracy(new_dico, top_matches)

fp_scores = all_fp_scores[positions, :]
k = 1000
top_matches = fp_scores.topk(k, 1, True)[1]
top_matches = top_matches[new_dico[:, 0], :]
results = get_topk_accuracy(new_dico, top_matches)
fp_top_matches = top_matches


recall(top_matches.flatten(), test_indices)
re2(top_matches, d)

fast_top_matches = []
for k in range(len(src_id)):
lst_k = fp_top_matches[k].numpy()
s = fast_scores[k][lst_k]
new_lst = lst_k[s.topk(10)[1].numpy()]
fast_top_matches.append(new_lst)

fast_top_matches = torch.from_numpy(np.asarray(fast_top_matches))
fast_top_matches = fast_top_matches[new_dico[:, 0], :]
new_results = get_topk_accuracy(new_dico, fast_top_matches)
print(new_results)


test_indices = test_dico[:, 1]

fast_top5 = torch.topk(fast_scores, 5, dim=1)


c = 0.7
x_fp = fp_quantile(src_fp, c)
fp_top_matches = []
alpha = 0.2
for k in range(len(src_id)):
lst_10 = fast_top5[1][k].numpy()
s_fast = fast_top5[0][k]
y_fp = tgt_fp[lst_10, :]
y_fp = fp_quantile(y_fp, c)
s_fp = x_fp[k] @ y_fp.T
s = s_fp + alpha * s_fast
new_lst = lst_10[s.topk(5)[1].numpy()]
fp_top_matches.append(new_lst)

fp_top_matches = torch.from_numpy(np.asarray(fp_top_matches))
fp_top_matches = fp_top_matches[new_dico[:, 0], :]
new_results = get_topk_accuracy(new_dico, fp_top_matches)
print(new_results)


X = {}
X['src'] = fast_embeddings['src'][src_id, :]
top5 = fast_scores.topk(5, 1, True)[1].numpy()
list_inds = list(set(top5.flatten()))
fast_ids = {}
for i in range(len(list_inds)):
    fast_ids[list_inds[i]] = i

X['tgt'] = fast_embeddings['tgt'][list_inds, :]

top1 = top5[:, 1].flatten()
inds = []
for i in range(len(top1)):
inds.append(fast_ids[top1[i]])


short_dico = []
for i in range(len(new_dico)):
    r = new_dico[i].numpy()
    if r[1] in fast_ids:
        short_dico.append([r[0], fast_ids[r[1]]])

A0 = X['src']
A1 = X['tgt'][inds, :]
W = robust_procrustes(A0, A1)
short_dico = torch.Tensor(np.asarray(short_dico)).type(torch.LongTensor)
for _ in range(15):  # refinement
X['src'] = X['src'] @ W.T
scores = cal_similarity('csls', None, X, 'fasttext', False) 
inds = align_words('nn', None, scores)
A0 = X['src'][inds[:, 0], :]
A1 = X['tgt'][inds[:, 1], :]
W = robust_procrustes(A0, A1)
scores = cal_similarity('csls', short_dico, X, 'fasttext', False) 
lst = align_words('nn', short_dico, scores)