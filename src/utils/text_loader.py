
import os, sys
import numpy as np
import configs


def load_vocabs_from_pairs(opts):
    fpath = opts.txt_dir + f'{opts.word_data}_{opts.src_lang}_{opts.tgt_lang}_{opts.data_mode}.txt'
    vocabs = {'src':[], 'tgt': []}
    with open(fpath) as f:
        lines = f.readlines()
    for l in lines:
        if configs.delimiters[opts.word_data] is not None:
            x, y = l.strip().lower().split()
        else:
            x, y = l.strip().lower().split()
        vocabs['src'].append(x)
        vocabs['tgt'].append(y)
    return vocabs

def load_vocabs(opts, lang):
    fpath = opts.txt_dir + f'{opts.word_data}_{lang}_{opts.data_mode}.txt'
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

def combine_files(opts):
    vocabs_src = load_vocabs(opts, opts.src_lang)
    vocabs_tgt = load_vocabs(opts, opts.tgt_lang)
    fpath = opts.txt_dir + f'{opts.word_data}_{opts.src_lang}_{opts.tgt_lang}_{opts.data_mode}.txt'
    f= open(fpath, "w") 
    for i in range(len(vocabs_src)):
        f.write(f"{vocabs_src[i]}{configs.delimiters[opts.word_data]}{vocabs_tgt[i]}\n")
    f.close()


def get_word2id(vocab):
    word2id = {}
    for i in range(len(vocab)):
        word2id[vocab[i]] = i
    return word2id
