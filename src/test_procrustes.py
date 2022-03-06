import torch
import numpy as np
import argparse
import random
import pdb
import os
from translation import robust_procrustes, robust_procrustes1, robust_procrustes2
from scipy.stats import ortho_group


def err(Y, X, T):
    Y = torch.from_numpy(Y)
    X = torch.from_numpy(X)
    if not torch.is_tensor(T):
        T = torch.from_numpy(T)
    
    return torch.norm(Y - X @ T)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    print("Seeded everything: {}".format(seed))


## Notation
# X: Normal distribution (n x k)
# T: random orthonormal matrix (k x k)
# Y = XT + E

#### set hyper-params
# n = 20,40
# k = 2,4,8
# p: proportion of error (0, 0.1, 0.5, 1)
# p_out: portion of outliers (0, 0.1, 0.2) = (0%, 10%, 20%)


parser = argparse.ArgumentParser(description='test procrustes')
parser.add_argument("-p", "--perturb_X", type=int, default=1)
parser.add_argument("-s", "--scheme_number", type=int, default=2)
args = parser.parse_args()
n, k = 20, 4 #10
if args.perturb_X:
    p, p_out, c = 0.1, 0.1, 0.001 #0.01 #0.5, 0.2, 0.01
else:
    p, p_out, c = 0, 0, 0



# 1. Load data X & ground-truth rotation matrix T
## generate clean X
set_seed(24)
X_cln = np.random.normal(size=(n,k))
print(np.around(X_cln, decimals=2))
T = ortho_group.rvs(dim=k) ## generate clean T

# 2. add error/outlier on the data X
# the errors on X were proportional to the standard deviations of the columns (X_pert = X_cln + X_err)
col_var = np.var(X_cln, axis=0).reshape(1,-1) # column-wise variance
X_err = c * np.repeat(col_var, n, axis=0)
X_pert = X_cln + X_err # error added

# outliers were created by choosing randomly p rows of X and multiplying them by -10
indices = np.arange(n)
np.random.shuffle(indices)
indices = indices[:int(n*p)]
X_pert[indices, :] = X_pert[indices, :] * (-10) # perturbation added
print(np.around(X_pert, decimals=2))
print((X_pert == X_cln).all())

# 3. generate observation Y from T, X_pert
Y = np.matmul(X_cln, T)
print("error (Y, X_cln, T_ground): ", err(Y, X_cln, T))
print("error (Y, X_pert, T_ground): ", err(Y, X_pert, T))


if args.scheme_number == 0:
    T_hat = robust_procrustes(torch.from_numpy(X_pert).float().cuda(), torch.from_numpy(Y).float().cuda())
    T_hat = T_hat.T # to make the result reasonable
elif args.scheme_number == 1:
    T_hat = robust_procrustes1(torch.from_numpy(X_pert).float().cuda(), torch.from_numpy(Y).float().cuda())
elif args.scheme_number == 2:
    T_hat = robust_procrustes2(torch.tensor(X_pert).float().cuda(), torch.tensor(Y).float().cuda(), torch.tensor(T).float().cuda())
else:
    raise NotImplementedError

print(T)
print(T_hat.cpu().numpy())

print("Perturb_X: {}, Scheme_number: {}".format(args.perturb_X, args.scheme_number))
print("error (Y, X_cln, T_hat): ", err(Y, X_cln, T_hat.cpu().double()))
print("error (Y, X_pert, T_hat): ", err(Y, X_pert, T_hat.cpu().double()))




