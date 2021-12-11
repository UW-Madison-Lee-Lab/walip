
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


def nearest_neighbor(embs):
    y = embs[0] @ embs[1].T 
    x = np.argmax(y, axis=1)
    return x, y


