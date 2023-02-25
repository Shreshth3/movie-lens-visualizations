# %%
from surprise import NMF
import numpy as np

# custom files
from surprise_nmf import get_U_V
from utils import read_data

data, movies = read_data()
model = NMF()
U, V = get_U_V(NMF, data)
# to get consistent with slides. We want U.T to be (M, k)
V = V.T
U = U.T

# M = 992, K = 15
# N = 1500, K = 15
# (992 users, 1500 movies)
print(U.shape)
print(V.shape)
Y = np.dot(U.T, V)

# %% 
A, S, B = np.linalg.svd(V)



# %%


print(A.shape, S.shape, B.shape)