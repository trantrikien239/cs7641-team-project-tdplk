from random import random
import numpy as np

class RandomGrader:
    def __init__(self):
        pass

    def fit(self, X, Y):
        return self

    def predict(self, X):
        N = X.shape[0]
        random_predict = np.random.randint(2,11, size=(N, 6)) / 2
        return random_predict
