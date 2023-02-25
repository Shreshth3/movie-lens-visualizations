"""
This model is trained using scikit-surprise.
This is the off-the-shelf implementation we chose to use.

Specifically, we use Non-negative Matrix Factorization.
"""
# %%
from surprise import Dataset, Reader, NMF, accuracy
from surprise.model_selection import train_test_split
from utils import read_data
import torch

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


def build_user_item_matrix(predictions):
    user_ids = list(map(lambda prediction: prediction.uid, predictions))
    item_ids = list(map(lambda prediction: prediction.iid, predictions))

    num_users = max(user_ids) + 1  # The ids are 0-indexed
    num_items = max(item_ids) + 1  # The ids are 0-indexed

    # TODO: think about default prediction
    user_item_matrix = torch.zeros((num_users, num_items))

    for prediction in predictions:
        user_id, item_id, estimated_rating = prediction.uid, prediction.iid, prediction.est

        user_item_matrix[user_id][item_id] = estimated_rating

    print(user_item_matrix)


# print(type(predictions[0]))
# print(accuracy.rmse(predictions))
build_user_item_matrix(predictions)

"""
Model accuracies (RMSE):
NMF: 0.8583
SVD: 0.8252
SVDpp: 0.8224
"""


# %%
