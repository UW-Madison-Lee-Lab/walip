
import os, sys
import numpy as np
import configs
from utils.helper import generate_path

def load_vocabs_from_pairs(langs, word_data, data_mode):
    fpath = generate_path('txt_pair', {'word_data': word_data, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})
    vocabs = {'src':[], 'tgt': []}
    with open(fpath) as f:
        lines = f.readlines()
    for l in lines:
        if configs.delimiters[word_data] is not None:
            x, y = l.strip().lower().split()
        else:
            x, y = l.strip().lower().split()
        vocabs['src'].append(x)
        vocabs['tgt'].append(y)
    return vocabs

def load_vocabs(lang, langs, word_data, data_mode):
    fpath = generate_path('txt_single', {'word_data': word_data, 'lang': lang, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})
    if not os.path.isfile(fpath):
        print("------> Error: Load vocabs", fpath, "file doesn't exist!!!")
        sys.exit('Done')
    with open(fpath) as f:
        lines = f.readlines()
    vocabs = [] # order
    for desc in lines:
        desc = desc.strip().lower()
        vocabs.append(desc)
    return vocabs

def write_vocabs(vocabs, lang, langs, word_data, data_mode):
    fpath = generate_path('txt_single', {'word_data': word_data, 'lang': lang, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})
    f = open(fpath, "w") 
    for i in range(len(vocabs)):
        f.write(f"{vocabs[i]}\n")
    f.close()

def combine_files(langs, word_data, data_mode):
    vocabs_src = load_vocabs(langs['src'], langs, word_data, data_mode)
    vocabs_tgt = load_vocabs(langs['tgt'], langs, word_data, data_mode)
    fpath = generate_path('txt_pair', {'word_data': word_data, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})
    f= open(fpath, "w") 
    for i in range(len(vocabs_src)):
        f.write(f"{vocabs_src[i]}{configs.delimiters[word_data]}{vocabs_tgt[i]}\n")
    f.close()


def get_word2id(vocab):
    word2id = {}
    for i in range(len(vocab)):
        if not (vocab[i] in  word2id):
            word2id[vocab[i]] = i
    return word2id
