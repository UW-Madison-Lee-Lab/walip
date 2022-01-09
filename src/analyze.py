


import torch
import numpy as np

# load the image embedding
# [N, D]
# calculate the similarity 
embs = {}
data = 'imagenet'

for i in range(2):
    emb_path = f'../dicts/images/{data}/image_feature_{data}_True_en{i}_k1.npy'
    emb = np.load(emb_path, allow_pickle=True)
    embs[i] = torch.Tensor(emb)

cos = torch.nn.CosineSimilarity(dim=1)
v = cos(embs[0], embs[1])
print(v)
from IPython import embed; embed()





