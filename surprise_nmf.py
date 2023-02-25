"""
This model is trained using scikit-surprise.
This is the off-the-shelf implementation we chose to use.

Specifically, we use Non-negative Matrix Factorization.
"""
from surprise import Dataset, Reader, NMF, accuracy
from surprise.model_selection import train_test_split
from utils import read_data

TRAIN_SET_SIZE = 109351
TEST_SET_SIZE = 12150

TEST_SIZE_PERCENTAGE = TEST_SET_SIZE / (TRAIN_SET_SIZE + TEST_SET_SIZE)

data, movies = read_data()


def get_U_V(model, data):
    """
    data: a pandas df
    """
    reader = Reader()
    dataset = Dataset.load_from_df(data, reader)

    train_data, _ = train_test_split(
        dataset, test_size=TEST_SIZE_PERCENTAGE)

    algo = model()
    algo.fit(train_data)

    U = algo.pu
    V = algo.qi

    return U, V


"""
Model accuracies (RMSE):
NMF: 0.8583
SVD: 0.8252
SVDpp: 0.8224
"""
