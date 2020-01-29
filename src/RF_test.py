from copy import deepcopy
from numpy import random
from joblib import Parallel, delayed
import pandas as pd


class RF:
    def __init__(self, model, numOfClf, n_jobs=None):
        self.model = model
        self.num = numOfClf
        self.task = model.task
        self.models = []
        self.n_jobs = n_jobs
        self.Alpha_list = []

    def fit(self, X, y, test_X=None, test_y=None, verbose=False):
        self.N = X.shape[0]
        self.X = X
        self.y = y
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single_model)() for i in range(self.num))
        for i in range(self.num):
            self.Alpha_list.append(self.models[i].alpha_list)
        df = pd.DataFrame(self.Alpha_list)
        return df

    def fit_single_model(self):
        N = self.N
        model_temp = deepcopy(self.model)
        bootstrap_index = random.randint(low=0, high=N, size=N)
        bootstrap_X = self.X[bootstrap_index]
        bootstrap_y = self.y[bootstrap_index]
        model_temp.fit(bootstrap_X, bootstrap_y)
        return model_temp
