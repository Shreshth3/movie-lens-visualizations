# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = 'data/data.csv'
MOVIES_PATH = 'data/movies.csv'

data = pd.read_csv(DATA_PATH, sep=',')
movies = pd.read_csv(MOVIES_PATH, sep=',')

# %%
ratings = data.loc[:, 'Rating']
ratings = ratings.to_numpy()

# %%
# All ratings in the MovieLens Dataset
plt.title(label="All ratings in the MovieLens Dataset")
plt.hist(ratings, bins=10)
plt.xlabel("Ratings")
plt.show()

# %%
# Ratings of 10 most popular movies

# %%
# Ratings of 10 best movies (movies w/ highest average rating)
grouped = data.groupby(by='Movie ID').mean()
# drop User ID
grouped = grouped.drop(columns='User ID')
# sort to get 10 most popular movies
print(grouped.sort_values('Rating', ascending=False))

# %%
# All ratings of movies from 3 genres of our choice (Action, Adventure, Animation)
# Can change:
GENRES_TO_KEEP = ['Action', 'Adventure', 'Animation']

TO_KEEP = ['Movie ID', 'Movie Title']

for GENRE in GENRES_TO_KEEP:
    temp = movies[TO_KEEP + [GENRE]]
    # get all the movies that is labeled the genre
    cleaned_movies = temp[(temp[GENRE] > 0)]
    # get the ids
    cleaned_ids = cleaned_movies['Movie ID'].to_numpy()
    # now grab the ratings
    cleaned_data = data.loc[data['Movie ID'].isin(cleaned_ids)]
    cleaned_ratings = cleaned_data.loc[:, 'Rating'].to_numpy()

    plt.title(label=f"All ratings for {GENRE}")
    plt.hist(cleaned_ratings, bins=10)
    plt.xlabel("Ratings")
    plt.show()

# get a thing with all three of them together, we don't need this XD
# cleaned_movies = temp[(temp[GENRES_TO_KEEP[0]] > 0) | (temp[GENRES_TO_KEEP[1]] > 0) | (temp[GENRES_TO_KEEP[2]] > 0)]
# cleaned_movies = temp[(temp[GENRES_TO_KEEP[0]] > 0)]

# %%
