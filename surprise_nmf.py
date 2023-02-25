"""
This model is trained using scikit-surprise.
This is the off-the-shelf implementation we chose to use.

Specifically, we use Non-negative Matrix Factorization.
"""
# %%
from surprise import Dataset, Reader, NMF, accuracy
from surprise.model_selection import train_test_split
from utils import read_data

TRAIN_SET_SIZE = 109351
TEST_SET_SIZE = 12150

TEST_SIZE_PERCENTAGE = TEST_SET_SIZE / (TRAIN_SET_SIZE + TEST_SET_SIZE)

data, movies = read_data()

reader = Reader()
dataset = Dataset.load_from_df(data, reader)

train_data, test_data = train_test_split(
    dataset, test_size=TEST_SIZE_PERCENTAGE)

algo = NMF()

algo.fit(train_data)

predictions = algo.test(test_data)

print(accuracy.rmse(predictions))


# %%
