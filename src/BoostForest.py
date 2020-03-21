from copy import deepcopy
from numpy import random
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from joblib import Parallel, delayed


class ASForest:
    def __init__(self, model, numOfClf, n_jobs=None):
        self.model = model
        self.num = numOfClf
        self.task = model.task
        self.models = []
        self.n_jobs = n_jobs
        if self.task == 'clf':
            self.prob = []
            self.EnsembleACC = []
            self.SingleACC = []
            # self.train_prob = []
        else:
            self.SingleLoss = []
            self.pred = []
            self.EnsembleLoss = []

    def fit(self, X, y, test_X=None, test_y=None, verbose=False):
        self.N = X.shape[0]
        self.X = X
        self.y = y
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single_model)() for i in range(self.num))
        for i in range(self.num):
            if self.task == 'clf':
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
            else:
                _, y_pred = self.models[i].predict_prob_output(test_X)
                single_loss_temp = mean_squared_error(test_y, y_pred)**0.5
                self.SingleLoss.append(single_loss_temp)
                self.pred.append(y_pred)
                pred_temp = np.array(self.pred).mean(axis=0)
                ensemble_loss_temp = mean_squared_error(test_y, pred_temp)**0.5
                self.EnsembleLoss.append(ensemble_loss_temp)
                if verbose:
                    print("Number of Base Learner: ", i + 1, "EnsembleLoss: ", ensemble_loss_temp,
                          "SingleLoss: ", single_loss_temp)

    def fit_single_model(self):
        N = self.N
        model_temp = deepcopy(self.model)
        bootstrap_index = random.randint(low=0, high=N, size=N)
        bootstrap_X = self.X[bootstrap_index]
        bootstrap_y = self.y[bootstrap_index]
        model_temp.fit(bootstrap_X, bootstrap_y)
        return model_temp
