import os, torch
import numpy as np
import scipy
from evals.word_translation import get_csls_word_translation, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy
from utils.loader import get_word2id, load_vocabs
from models.ops import load_embedding
import argparse
import configs

os.environ['TOKENIZERS_PARALLELISM'] = "false"

# main
parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("-w", "--word_data", type=str, default="cifar100", help="Dataset for word translation")
parser.add_argument("-i", "--image_data", type=str, default="cifar100", help="Image dataset for fingerprint")
parser.add_argument("-l", "--learning_mode", type=str, default="unsupervised", help="learning method: unsupervised or supervised")
parser.add_argument("-m", "--matching_method", type=str, default="lin_asm", help="biparite matching method [nn, lin_asm]")
parser.add_argument("-s", "--sim_score", type=str, default="cosine", help="similarity score [cosine, csls, inner_prod]")
parser.add_argument("-t", "--data_mode", type=str, default="test", help="mode of evaluation")
parser.add_argument("-e", "--emb_type", type=str, default="fp", help="type of embedding: fingerprint, cliptext, fasttext")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='it', help="Target language")
# parse parameters
params = parser.parse_args()
  
        



def process_configs(params):
    dict_pth = '../dicts/'
    configs.paths['emb_dir'] = dict_pth + 'embeddings/'
    configs.paths['img_dir'] = dict_pth + 'images/{}/'.format(params.image_data)
    configs.paths['txt_dir'] = dict_pth + 'texts/{}/'.format(params.word_data)

    configs.langs['src'] = params.src_lang
    configs.langs['tgt'] = params.tgt_lang

# prepare embedding
def prepare_embeddings(params):
    vocabs, word2ids, embs = {}, {}, {}
    for l in ['src', 'tgt']:
        vocabs[l] = load_vocabs(params.word_data, configs.langs[l], params.data_mode)
        word2ids[l] = get_word2id(vocabs[l])
        embs[l] = load_embedding(params.emb_type, params.word_data, params.image_data, configs.langs[l], vocabs[l], params.data_mode)
    return word2ids, embs


process_configs(params)
word2ids, embs = prepare_embeddings(params)
test_fpath = configs.paths['txt_dir'] + '{}_{}_{}_{}.txt'.format(params.word_data, params.src_lang, params.tgt_lang, params.data_mode)
dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'])

print('==== Eval', params.word_data, ':', params.emb_type, params.sim_score, params.matching_method)
if params.learning_mode == 'supervised':
    def train_supervision(X1, X2):
        X = X1.T @ X2
        # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
        U, Sigma, VT = scipy.linalg.svd(X, full_matrices=True)
        W = U @ VT
        return W

    W = train_supervision(embs['src'], embs['tgt'])
    # evaluate test
    # embs_0 = embs[0] @ W.T
    # load test
    embs['src'] = embs['src'] @ W.T

# Testing 
if params.sim_score in ['csls', 'cosine']:
    dico = dico.cuda()
    for s in ['src', 'tgt']:
        embs[s] = torch.from_numpy(embs[s]).cuda()
        # assert dico[:, i].max() < embs[s].size(0)
    scores = get_csls_word_translation(dico, embs['src'], embs['tgt'], params.sim_score)
elif params.sim_score == 'inner_prod':
    test_emb0 = embs['src'][dico[:, 0]]
    test_emb1 = embs['src'][dico[:, 1]]
    scores = test_emb0 @ test_emb1.T 

if params.matching_method == 'nn':
    results = get_topk_translation_accuracy(dico, scores)
    print(results)
elif params.matching_method == 'hungarian':
    cost = -scores.cpu().numpy()
    dico = dico.cpu()    
    _, col_ind = scipy.optimize.linear_sum_assignment(cost)
    acc, wrong_pairs = get_accuracy(dico, col_ind)