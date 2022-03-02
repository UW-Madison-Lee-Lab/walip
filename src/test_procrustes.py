import torch
import numpy as np
import pdb
from translation import robust_procrustes, robust_procrustes1
from scipy.stats import ortho_group

## Notation
# X: Normal distribution (n x k)
# T: random orthonormal matrix (k x k)
# Y = XT + E

#### set hyper-params
# n = 20,40
# k = 2,4,8
# p: proportion of error (0, 0.1, 0.5, 1)
# p_out: portion of outliers (0, 0.1, 0.2) = (0%, 10%, 20%)
perturb_X = False
scheme_number = 0


n, k = 20, 4
if perturb_X:
    p, p_out, c = 0.1, 0.1, 0.01 #0.5, 0.2, 0.01
else:
    p, p_out, c = 0, 0, 0

# 1. Load data X & ground-truth rotation matrix T
## generate clean X
X_cln = np.random.normal(size=(n,k))
print(np.around(X_cln, decimals=2))
## generate clean T
T = ortho_group.rvs(dim=k)

# 2. add error/outlier on the data X
# the errors on X were proportional to the standard deviations of the columns (X_pert = X_cln + X_err)
col_var = np.var(X_cln, axis=0).reshape(1,-1) # column-wise variance
X_err = c * np.repeat(col_var, 20, axis=0)
X_pert = X_cln + X_err # error added

# outliers were created by choosing randomly p rows of X and multiplying them by -10
indices = np.arange(n)
np.random.shuffle(indices)
indices = indices[:int(n*p)]
X_pert[indices, :] = X_pert[indices, :] * (-10) # perturbation added
print(np.around(X_pert, decimals=2))
print((X_pert == X_cln).all())

# 3. generate observation Y from T, X_pert
Y = np.matmul(X_pert, T)

if scheme_number == 0:
    T_hat = robust_procrustes(torch.from_numpy(X_pert).float().cuda(), torch.from_numpy(Y).float().cuda())
    T_hat = T_hat.T
elif scheme_number == 1:
    T_hat = robust_procrustes1(torch.from_numpy(X_pert).float().cuda(), torch.from_numpy(Y).float().cuda())
else:
    raise NotImplementedError

print(T)
print(T_hat)

#pdb.set_trace()
print(np.linalg.norm(Y - np.matmul(X_cln, T_hat.cpu().numpy())))
print(np.linalg.norm(Y - np.matmul(X_pert, T_hat.cpu().numpy())))


