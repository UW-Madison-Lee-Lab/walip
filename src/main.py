import os, torch
import numpy as np
from models.uwt import linear_assigment, nearest_neighbor
from evals.word_translation import get_csls_word_translation, get_topk_translation_accuracy, load_dictionary
from utils.helper import get_accuracy
from utils.loader import get_word2id
from utils.loader import load_vocab_translation
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
parser.add_argument("-m", "--method", type=str, default="csls_knn_10", help="method for dictionary generation [nn, csls, matching]")

parser.add_argument("-t", "--test", type=str, default="test", help="mode of evaluation")

parser.add_argument("-e", "--emb_type", type=str, default="fp", help="type of embedding: fingerprint, cliptext, fasttext")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='it', help="Target language")
# parse parameters
params = parser.parse_args()
  
        

mode = params.test
configs.langs['src'] = params.src_lang
configs.langs['tgt'] = params.tgt_lang

# prepare embedding
vocabs, translation = load_vocab_translation(params.word_data, [params.src_lang, params.tgt_lang], mode)
embs = {}
for i, lang in enumerate([params.src_lang, params.tgt_lang]):
    embs[i] = load_embedding(params.emb_type, params.word_data, params.image_data, lang, vocabs[i], mode)

print('Eval', params.word_data, 'with', params.method, params.emb_type)
# perform translation
if params.method in ['nn', 'csls_knn_10']:
    # test data
    dico_eval = '../dicts/texts/{}/{}_{}_{}_test.txt'.format(params.word_data, params.word_data, params.src_lang, params.tgt_lang)
    word2ids = get_word2id(vocabs)
    dico = load_dictionary(dico_eval, word2ids[0], word2ids[1])
    dico = dico.cuda()
    for i in range(2):
        embs[i] = torch.from_numpy(embs[i]).cuda()
        assert dico[:, i].max() < embs[i].size(0)
    # translation
    scores = get_csls_word_translation(dico, embs, method=params.method)
    results = get_topk_translation_accuracy(scores, dico)
    print(results)
elif params.method == 'matching':
    col_ind = linear_assigment(embs)
    acc = get_accuracy(vocabs, translation, col_ind)
    print(acc)
elif params.method == '1nn':
    col_ind, scores = nearest_neighbor(embs)
    acc = get_accuracy(vocabs, translation, col_ind)
    print(acc)
    from IPython import embed; embed()
