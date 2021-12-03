from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy
from ops import *
from helper import *
from templates import templates, generate_texts
# import sys
# sys.path.append('../')
from word_translation import get_word_translation_accuracy, read_txt_embeddings
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

import pdb

def example():
    eng_vocab = ["outbreak",
         "legendary",
         "handball",
         "georgian",
         "copenhagen",
    ]
    eng_texts = [f"This is a photo of a " + desc for desc in eng_vocab]
    eng_params = {
        'clip_name': "ViT-B/32"
    }
    
    # italian
    ita_params = {
        'clip_name': 'ViT-B/32',
    }
    ita_vocab = [
        'epidemia',
        'leggendario',
        'pallamano',
        'georgiano',
        'copenhagen',
        ]
    ita_vocab.reverse()

    ita_texts = [f"Questa è una foto di a "+ desc for desc in ita_vocab]

    probs_EN = get_fingerprints(eng_texts, eng_params['clip_name'], is_clip=True, image_name='tiny', num_images=1)
    probs_FR = get_fingerprints(ita_texts, ita_params['clip_name'], is_clip=False, image_name='tiny', num_images=1)

    cost = -(probs_EN @ probs_FR.T)
    _, col_ind = linear_sum_assignment(cost)
    print(col_ind)


def load_vocab_translation(names, langs=['en','it'], mode='test'):
    if names['data'] in ['dict']:
        vocabs, translation = load_data_from_one_files(names['data'], langs, mode)
    else:
        vocabs, translation = load_data_from_two_files(names['data'], langs) #, mode)
    return vocabs, translation

def load_embedding(names, langs=['en','it'], k=-1, mode='test'):
    vocabs, translation = load_vocab_translation(names, langs)
    embs = {} # types of e
    for i in range(2):
        emb_path = '../dicts/npy/{}_{}_embedding_{}_{}.npy'.format(names['data'], names['image'], langs[i], mode)
        # emb_path = '../dicts/npy/fp_cifar100_{}.npy'.format(langs[i])
        if os.path.isfile(emb_path):
            print('load ', emb_path)
            embs[i] = np.load(emb_path, allow_pickle=True)
        else:
            text_emb_path = '../dicts/npy/{}_text_embedding_{}_{}_{}.npy'.format(names['data'], langs[i], k, mode)
            if os.path.isfile(text_emb_path):
                text_features = np.load(text_emb_path, allow_pickle=True)
                text_features = torch.Tensor(text_features)
                clip_model, preprocess = clip.load("ViT-B/32")
                clip_model.cuda().eval()
                image_features = get_clip_image_features(clip_names[langs[i]], names['image'], 1, clip_model)
                logit_scale = clip_model.logit_scale.exp().float()
                embs[i] = cal_probs_from_features(image_features, text_features, logit_scale)
            else:
                texts = generate_texts(templates[langs[i]], vocabs[i])
                embs[i] = get_fingerprints(texts, langs[i], image_name=names['image'], num_images=1, K=len(templates[langs[i]]))
            np.save(emb_path, embs[i]) 
    return embs, vocabs, translation


def translate(names, langs=['en','it']):
    embs, vocabs, translation = load_embedding(names, langs)
    y = embs[0] @ embs[1].T 
    cost = -y
    _, col_ind = linear_sum_assignment(cost)
    s = 0
    for i in range(len(vocabs[0])):
        if vocabs[1][col_ind[i]] in translation[vocabs[0][i]]:
            s+=1
    print('Accuracy: {:.4f}'.format(s/len(vocabs[0])))


def get_accuracy(names, langs=['en','it']):
    embs, vocabs, _ = load_embedding(names, langs, k=1)
    # vocabs, translation = load_vocab_translation(names, langs)
    # embs = {}
    # embs[0] = np.load('../dicts/npy/fasttext_en_test.npy', allow_pickle=True)
    # embs[1] = np.load('../dicts/npy/fasttext_it_test.npy', allow_pickle=True)
    for i in range(2):
        embs[i] = torch.Tensor(embs[i]).cuda()
    
    print('word2id')
    word2ids = {0:{}, 1:{}}
    for k in range(2):
        for i in range(len(vocabs[k])):
            word2ids[k][vocabs[k][i]] = i

    print('accuracy')
    method = 'csls_knn_10'
    DIC_EVAL_PATH = '../dicts/texts/'
    dico_eval = os.path.join(DIC_EVAL_PATH, '{}_{}_{}_test.txt'.format(names['data'], langs[0], langs[1]))
    results = get_word_translation_accuracy(langs[0], word2ids[0], embs[0], langs[1], word2ids[1], embs[1], method, dico_eval)
    print(results)



def load_text_embedding(names, langs=['en','it'], k=-1, mode='test'):
    vocabs, translation = load_vocab_translation(names, langs, mode)
    # embs = {} # types of e
    # for i in range(2):
    #     embs[i] = np.load('../dicts/npy/fasttext_{}_{}.npy'.format(langs[i], mode))
    # return embs, vocabs, translation   
    for i in range(2):
        emb_path = '../dicts/npy/{}_text_embedding_{}_{}_{}.npy'.format(names['data'], langs[i], k, mode)
        if os.path.isfile(emb_path):
            print('load ', emb_path)
            embs[i] = np.load(emb_path, allow_pickle=True)
        else:
            texts = generate_texts(templates[langs[i]], vocabs[i], k=k)
            clip_embeds = []
            K = len(templates[langs[i]]) if k == -1 else 1
            bz = 1024 * K
            for t in tqdm(chunks(bz, texts)):
                em = get_fingerprints(t, langs[i], image_name=names['image'], num_images=1, K=K, text_embedding=True) 
                clip_embeds.append(em.cpu().numpy())
            embs[i] = np.concatenate(clip_embeds, axis=0)
            np.save(emb_path, embs[i]) 
    return embs, vocabs, translation

def supervised(names, langs):
    from sklearn.utils.extmath import randomized_svd

    def train(embs, vocabs, translation):
        X = embs[1].T @ embs[0] 
        # U, Sigma, VT = randomized_svd(X, n_components=1000, n_iter=5, random_state=42)
        U, Sigma, VT = scipy.linalg.svd(X, full_matrices=True)
        W = U @ VT
        # from matplotlib import pyplot as plt
        # plt.imshow(W)
        # plt.colorbar()
        # plt.savefig('../results/W_fasttext.png')
        embs_0 = embs[0] @ W.T
        y = embs_0 @ embs[1].T
        cost = -y
        _, col_ind = linear_sum_assignment(cost)
        s = 0
        for i in range(len(vocabs[0])):
            if vocabs[1][col_ind[i]] in translation[vocabs[0][i]]:
                s+=1
        print('Training accuracy: {:.4f}'.format(s/len(vocabs[0])))
        return W

    embs, vocabs, translation = load_text_embedding(names, langs, 1, 'val')
    # embs[0] = embs[0][:1000, :]
    # embs[1] = embs[1][:1000, :]

    W = train(embs, vocabs, translation)
    # test
    embs, word2ids = {}, {}
    # embs, vocabs, translation = load_text_embedding(names, langs, 1, 'test')
    for i in range(2):
        emb_path = '../muse/data/wiki.{}.vec'.format(langs[i])
        id2word, word2id, embeddings = read_txt_embeddings(langs[i], emb_path)
        embs[i] = embeddings
        word2ids[i] = word2id
        print("Load 200k data from {}".format(langs[i]))

    embs[0] = embs[0] @ W.T
    for i in range(2):
        embs[i] = torch.Tensor(embs[i]).cuda()
    
    method = 'csls_knn_10'
    results = get_word_translation_accuracy(langs[0], word2ids[0], embs[0], langs[1], word2ids[1], embs[1], method, dico_eval='default')
    print(results)

        
        

if __name__ == '__main__':
    #names = {'data': 'noun', 'image': 'cifar100'}
    names = {'data': 'cifar100', 'image': 'cifar100'}
    langs = ['en', 'it']
    # get_accuracy(names, langs)
    # translate(names, langs)
    # supervised(names, langs)
    vocabs, translation = load_data_from_two_files(names['data'], langs, 'test')
    f= open('../dicts/texts/noun_en_it_test.txt', "w") 
    for i in range(len(vocabs[0])):
        f.write("{} {}\n".format(vocabs[0][i], vocabs[1][i]))
    f.close()


   
