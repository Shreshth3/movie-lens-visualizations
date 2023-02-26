# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import *

# custom files
from bias_model import *
from utils import read_data
data, movies = read_data()
U, V, a, b = get_U_V_a_b(data)

num_users, num_movies = compute_num_users_and_movies(data)
eta = 0.03
reg = 0.1
eps = 0.0001
max_epochs = 300
data_as_numpy = data.to_numpy(copy=True).astype(int)

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
  ax.text(x+.01, y+.01, str(first_10[i]))

# %%
# just grab first 10 movies
# TODO: abstract this out so we pick 10 actually useful movies, like use indices
to_viz_np = []
movie_titles = movies['Movie Title']
besttitles = []
for i in bestmovies:
    to_viz_np.append(approx_V[:,i])
    besttitles.append(movie_titles[i])
to_viz_np = np.array(to_viz_np)
to_viz_df = pd.DataFrame(to_viz_np, columns = ['x', 'y'])
# plot the points.
ax = sns.scatterplot(to_viz_df, x='x', y='y')

# TODO: if update above, then update below
first_10 = movie_titles[:10]

for i, (x, y) in to_viz_df.iterrows():
  # TODO: also might need to change the first_10[i] here too.
  ax.text(x+.01, y+.01, str(besttitles[i]))
# %%
