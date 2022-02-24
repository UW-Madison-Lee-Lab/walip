from utils.text_loader import load_vocabs, load_vocabs_from_pairs, write_vocabs, combine_files
from utils.helper import generate_path
import numpy as np

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
langs = {'src': 'it', 'tgt': 'en'}
word_data = 'wiki'
data_mode = 'test'
vocabs = load_vocabs_from_pairs(langs, word_data, data_mode)
for l, lang in langs.items():
    write_vocabs(vocabs[l], langs[l], langs, word_data, data_mode)
    print('Done', lang)

# combine_files(langs, word_data, data_mode)