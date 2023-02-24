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
ratings_for_top_ten_movies = top_ten_movies['Rating'].to_numpy()

plt.title(label="Ratings for top 10 most popular movies")
plt.hist(ratings, bins=10)
plt.ylabel("Num movies")
plt.xlabel("Ratings")
plt.show()


# %%
# Ratings of 10 best movies
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
plt.hist(top_ratings, bins=10)
plt.xlabel("Ratings")
plt.title(label="All Ratings for Top 10 Rated Movies")
plt.show()



# %%

# %%

# %%

# %%

# %%
# All ratings of movies from 3 genres of our choice (Action, Adventure, Animation)
data2 = pd.DataFrame.copy(data, deep=True)
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


# %%
