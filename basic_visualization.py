# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data/data.csv'
MOVIES_PATH = 'data/movies.csv'

data = pd.read_csv(DATA_PATH, sep=',')
movies = pd.read_csv(MOVIES_PATH, sep=',')

sns.set_theme(style="darkgrid")
# %%
def plot_hist(title, data):
    # use the to_numpy() if using matplotlib
    # data = data.to_numpy()

    plt.title(label=title)
    # plt.hist(data, bins=10)
    sns.histplot(data, bins=10)
    plt.xlim(0.5, 5)
    # plt.xlabel("Ratings")
    # plt.ylabel("Num movies")
    plt.show()

# %%
# from collections import defaultdict
# counter = defaultdict(int)

# # manually go through all the data again
# for index, (user, movie, rating) in data.iterrows():
#     # print(user, movie, rating)
#     counter[rating] += 1

# print(counter)

# %%
# All ratings in the MovieLens Dataset
ratings = data.loc[:, 'Rating']

plot_hist("All ratings in the MovieLens Dataset", ratings)

# %%
# Ratings of 10 most popular movies

def get_top_ten_most_popular_movie_ids():
    movies_with_frequencies = data.groupby(by='Movie ID').count()
    movies_by_frequency = movies_with_frequencies.sort_values(
        'Rating', ascending=False)

    top_ten_movies_by_frequency = movies_by_frequency[:10]
    top_ten_movie_ids_by_frequency = top_ten_movies_by_frequency.index.values

    return top_ten_movie_ids_by_frequency


top_ten_movie_ids_by_frequency = get_top_ten_most_popular_movie_ids()

top_ten_movies = data.loc[data['Movie ID'].isin(
    top_ten_movie_ids_by_frequency)]
ratings_for_top_ten_movies = top_ten_movies['Rating']

plot_hist("Ratings for top 10 most popular movies", ratings_for_top_ten_movies)

# %%
# Ratings of 10 best movies (movies w/ highest average rating)
grouped = data.groupby(by='Movie ID').mean()
# drop User ID
grouped = grouped.drop(columns='User ID')
# sort to get 10 most popular movies
top_10_movies = grouped.sort_values('Rating', ascending=False)[:10]
top_10_movie_IDs = top_10_movies.index.values
# print(top_10_movie_IDs)

top_ratings = []
for mID in top_10_movie_IDs:
    mov_ratings = data.loc[data["Movie ID"] == mID]
    just_ratings = mov_ratings.to_numpy()[:,2]
    top_ratings.extend(just_ratings)
    # break
    # for person in mov_ratings:
    #     print(person)
    #     break
    #     top_ratings.append(rat)
    # break
print(len(top_ratings))

plot_hist("All Ratings for Top 10 Rated Movies", top_ratings)

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
    cleaned_ratings = cleaned_data.loc[:, 'Rating']

    plot_hist(f"All ratings for {GENRE}", cleaned_ratings)

# get a thing with all three of them together, we don't need this XD
# cleaned_movies = temp[(temp[GENRES_TO_KEEP[0]] > 0) | (temp[GENRES_TO_KEEP[1]] > 0) | (temp[GENRES_TO_KEEP[2]] > 0)]
# cleaned_movies = temp[(temp[GENRES_TO_KEEP[0]] > 0)]

# %%
