import numpy as np
import pandas as pd

DATA_PATH = 'data/data.csv'
MOVIES_PATH = 'data/movies.csv'
K = 20

data = pd.read_csv(DATA_PATH, sep=',')
movies = pd.read_csv(MOVIES_PATH, sep=',')


def read_data():
    data = pd.read_csv(DATA_PATH, sep=',')
    movies = pd.read_csv(MOVIES_PATH, sep=',')

    return data, movies
