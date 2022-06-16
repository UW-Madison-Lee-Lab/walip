from utils.text_loader import load_vocabs, load_vocabs_from_pairs, write_vocabs, combine_files, get_word2id
from utils.helper import generate_path
import numpy as np
# from nltk.corpus import wordnet as wn
import spacy
from evals.word_translation import load_dictionary
import configs 

def check_noun(words):
    pos_all = dict()
    for w in words:
        pos_l = set()
        for tmp in wn.synsets(w):
            if tmp.name().split('.')[0] == w:
                pos_l.add(tmp.pos())
        pos_all[w] = pos_l
    return pos_all

# noun_vocabs = load_vocabs_from_pairs('en', 'ru', 'noun', 'test1')
# wiki_vocabs = load_vocabs('en', 'wiki', 'full')

# fpath = generate_path('txt_pair', {'word_data': 'noun', 'src_lang': 'en', 'tgt_lang': 'ru', 'data_mode': 'test'})

# dico = np.load('../dicts/texts/noun/dico_en_ru.npy', allow_pickle=True)
# lst_en_vocabs = [wiki_vocabs[k] for k in dico[:, 0]]

# f = open(fpath, 'w')
# count = 0
# for i in range(len(noun_vocabs['src'])):
#     if noun_vocabs['src'][i] in lst_en_vocabs:
#         count += 1
#         f.write(f"{noun_vocabs['src'][i]} {noun_vocabs['tgt'][i]}\n")
# f.close()
# print('count:', count)
langs = {'src': 'en', 'tgt': 'fr'}
word_data = 'wiki'
data_mode = 'test'
# emb_type = 'fasttext'
# lang = langs['src']

# vocabs = {}
# for key, lang in langs.items():
    # vocabs[key] = load_vocabs(lang, langs, word_data, data_mode)
# fpath = generate_path('emb_fasttext', {'word_data': word_data, 'lang': lang, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})
# embs = np.load(fpath, allow_pickle=True)
# v = []
# inds = []
# for i in range(len(vocabs)):
#     if not(vocabs[i] in v):
#         v.append(vocabs[i])
#         inds.append(i)
# print(len(inds))
# indices = np.asarray(inds)
# np.random.shuffle(indices)
# v_new = [vocabs[i] for i in indices]
# write_vocabs(v_new, lang, langs, word_data, data_mode)
# np.save(fpath, embs[indices, :])
# indices = np.load('../results/indices_en_ru_wiki_noun_True.npy', allow_pickle=True)
# nlp = spacy.load("en_core_web_sm")
# count = 0
# for w in vocabs:
#     doc = nlp(w)
#     if doc[0].pos_ == 'NOUN':
#         count += 1
# print('Nouns: ', count/len(vocabs))

# Test-dico
# word_data = 'wiki'
# word2ids = {}
# for key, lang in langs.items():
#     vocab = load_vocabs(lang, langs, word_data, data_mode)
#     word2ids[key] = get_word2id(vocab)

# test_fpath = generate_path('txt_pair', {'word_data': word_data, 'src_lang': langs['src'], 'tgt_lang': langs['tgt'], 'data_mode': data_mode})

# test_dico = load_dictionary(test_fpath, word2ids['src'], word2ids['tgt'], delimiter=configs.delimiters[word_data])

# d = {}
# for i in range(len(test_dico)):
#     r = test_dico[i, :].numpy()
#     if r[0] in d:
#         d[r[0]].append(r[1])
#     else:
#         d[r[0]] = [r[1]]

# id_tgt = [word2ids['tgt'][w] for w in vocabs['tgt']]
# id_src = [word2ids['src'][w] for w in vocabs['src']]
# count = 0
# from IPython import embed; embed()
# for id0 in id_src:
#     for i in d[id0]:
#         if i in id_tgt:
#             count += 1
#             break
# print('Acc:', count/len(vocabs)*100)


vocabs = load_vocabs_from_pairs(langs, word_data, data_mode)
for l, lang in langs.items():
    write_vocabs(vocabs[l], langs[l], langs, word_data, data_mode)
    print('Done', lang)

# combine_files(langs, word_data, data_mode)