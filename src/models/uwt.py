
import numpy as np
from scipy.optimize import linear_sum_assignment
from evals.word_translation import load_dictionary, get_csls_word_translation

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


def linear_assigment(embs):
    y = embs[0] @ embs[1].T 
    cost = -y
    _, col_ind = linear_sum_assignment(cost)
    return col_ind  

def nearest_neighbor(embs):
    y = embs[0] @ embs[1].T 
    x = np.argmax(y, axis=1)
    return x

def supervised(names, langs):
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
    # embs, word2ids = {}, {}
    embs, vocabs, translation = load_text_embedding(names, langs, 1, 'test')
    # for i in range(2):
    #     emb_path = '../muse/data/wiki.{}.vec'.format(langs[i])
    #     id2word, word2id, embeddings = read_txt_embeddings(langs[i], emb_path)
    #     embs[i] = embeddings
    #     word2ids[i] = word2id
    #     print("Load 200k data from {}".format(langs[i]))

    print('word2id')
    word2ids = {0:{}, 1:{}}
    for k in range(2):
        for i in range(len(vocabs[k])):
            word2ids[k][vocabs[k][i]] = i

    embs[0] = embs[0] @ W.T
    for i in range(2):
        embs[i] = torch.Tensor(embs[i]).cuda()
    
    method = 'csls_knn_10'
    results = get_word_translation_accuracy(langs[0], word2ids[0], embs[0], langs[1], word2ids[1], embs[1], method, dico_eval='default')
    print(results)

      