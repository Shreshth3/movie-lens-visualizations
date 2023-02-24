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

#%%
# Ratings of 10 most popular movies
# plt.title(label="All ratings in the MovieLens Dataset")
# plt.hist(ratings, bins=10)
# plt.show()
# data.sort_values('Movie ID')
print(data)

# %%
# Ratings of 10 best movies
# all_movie_ids = data.loc[:, 'Movie ID'].to_numpy()
# movie_ids = np.unique(all_movie_ids)

# def get_avg_rating(data, movie_id):
#     data_for_movie = data.loc[data['Movie ID'] == movie_id]

#     ratings = data_for_movie[:, 2]

#     return np.mean(ratings)

# movie_ids_with_avg_rating = []

# for movie_id in movie_ids:
#     avg_rating = get_avg_rating(data, movie_id)

#     movie_ids_with_avg_rating.append((movie_id, avg_rating))

# movie_ids_with_avg_rating.sort(key=lambda id, rating: rating)
# ten_best_movies_with_ratings = movie_ids_with_avg_rating[:10]

# ten_best_movie_ids = ten_best_movies_with_ratings[:, 0]
# ten_best_movie_avg_ratings = ten_best_movies_with_ratings[:, 1]

# plt.title(label="Avg ratings for 10 best movies")
# plt.scatter(ten_best_movie_ids, ten_best_movie_avg_ratings)
# plt.xlabel("Top 10 movies")
# plt.ylabel("Avg rating")
# plt.show()
    

#%%
# All ratings of movies from 3 genres of our choice (Action, Adventure, Animation)
data2 = pd.DataFrame.copy(data, deep = True)
data2 = data2.to_numpy()

top3_ratings = []

for point in data2:
    uid, mid, rating = point
    movie = movies.loc[data["Movie ID"] == mid]
    movie = movie.to_numpy()
    # print(movie)
    # break
    # if movie.any(movie[0][3] == 1 or movie[0][4] == 1 or movie[0][5] == 1):
    #     top3_ratings.append(rating)
    if len(movie) == 0:
        continue
    if (movie[0][3] == 1 or movie[0][4] == 1 or movie[0][5] == 1):
        top3_ratings.append(rating)

print(f"TOP3 RATING LENGTH: {len(top3_ratings)}")


# only_wanted_genres = data.loc[data['Action'] == 1 or data['Adventure'] == 1 or data['Animation'] == 1]
# partial_ratings = data.loc[:, 'Rating']
# print(f"LENGTH OF THE THING IS: {len(partial_ratings)}")


#%%


