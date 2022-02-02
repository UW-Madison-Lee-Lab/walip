from umap import UMAP
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

W = np.load('../../dicts/best_W.npy', allow_pickle=True)
X = np.load('../../dicts/embeddings/fasttext_cifar100_en_test.npy', allow_pickle=True)
Y = np.load('../../dicts/embeddings/fasttext_cifar100_en_test.npy', allow_pickle=True)

X = X @ W.T
data = np.concatenate([X, Y], axis=0)
# tsne = TSNE(n_components=3, random_state=0)
#  projections = tsne.fit_transform(data)
umap_2d = UMAP(random_state=0)
proj_2d = umap_2d.fit_transform(data)

fig = plt.figure()
X_2d = proj_2d[:45, ...]
Y_2d = proj_2d[45:, ...]
plt.scatter(X_2d[:, 0], X_2d[:, 1],  marker='.', c='r')
plt.scatter(Y_2d[:, 0], Y_2d[:, 1],  marker='.', c='b')
plt.savefig('../../results/plots/embs_tn.png')