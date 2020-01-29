from copy import deepcopy
from numpy import random
from sklearn.metrics import accuracy_score
import numpy as np
from joblib import Parallel, delayed


class ASForest_LMT:
    def __init__(self, model, numOfClf, n_jobs=None):
        self.model = model
        self.num = numOfClf
        self.models = []
        self.n_jobs = n_jobs
        self.prob = []
        self.EnsembleACC = []
        self.SingleACC = []

    def fit(self, X, y, test_X=None, test_y=None, verbose=False):
        self.N = X.shape[0]
        self.X = X
        self.y = y
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single_model)() for i in range(self.num))
        for i in range(self.num):
            # Single Learner
            prob_temp, _ = self.models[i].predict_prob_output(test_X)
            y_pred = self.models[i].prob2pred_label(prob_temp)
            single_acc_temp = accuracy_score(test_y, y_pred)
            self.SingleACC.append(single_acc_temp)
            # Ensemble Learner
            self.prob.append(prob_temp)
            prob_temp = np.array(self.prob).mean(axis=0)
            y_pred = self.models[i].prob2pred_label(prob_temp)
            ensemble_acc_temp = accuracy_score(test_y, y_pred)
            self.EnsembleACC.append(ensemble_acc_temp)
            if verbose:
                print("Number of Base Learner: ", i + 1, "EnsembleACC: ", ensemble_acc_temp,
                      "SingleACC: ", single_acc_temp)

    def fit_single_model(self):
        N = self.N
        model_temp = deepcopy(self.model)
        bootstrap_index = random.randint(low=0, high=N, size=N)
        bootstrap_X = self.X[bootstrap_index]
        bootstrap_y = self.y[bootstrap_index]
        model_temp.fit(bootstrap_X, bootstrap_y)
        return model_temp
