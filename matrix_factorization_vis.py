# %%
from surprise import NMF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# custom files
from surprise_nmf import get_U_V
from utils import read_data

data, movies = read_data()
model = NMF()
U, V = get_U_V(NMF, data)

# M = 992, K = 15
# N = 1500, K = 15
# Y = (992 users, 1500 movies)
Y = np.dot(U.T, V)

# %% 
A, S, B = np.linalg.svd(V)
approx_V = np.dot(A[:, :2].T, V)
approx_U = np.dot(A[:, :2].T, U)

# %%
# just grab first 10 movies
# TODO: abstract this out so we pick 10 actually useful movies, like use indices
to_viz_np = approx_V[:, :10]
to_viz_df = pd.DataFrame(to_viz_np.T, columns = ['x', 'y'])

# plot the points.
ax = sns.scatterplot(to_viz_df, x='x', y='y')

movie_titles = movies['Movie Title']
# TODO: if update above, then update below
first_10 = movie_titles[:10]

for i, (x, y) in to_viz_df.iterrows():
  # TODO: also might need to change the first_10[i] here too.
  ax.text(x+.005, y, str(first_10[i]))

# %%