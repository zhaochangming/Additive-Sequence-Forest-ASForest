from sklearn.linear_model import LinearRegression
import numpy as np


class SingleLinearRegression:
    def __init__(self):
        self.model = None
        self.attributeIndex = None
        self.yMean = None

    def fit(self, X, y, weight):
        numAttributes = X.shape[1]
        lossList = []
        modelList = []
        for i in range(numAttributes):
            X_temp = X[:, i]
            X_temp = X_temp.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X_temp, y, weight)
            lossList.append(lr.score(X_temp, y, weight))
            modelList.append(lr)
        best_r2_score = max(lossList)
        if best_r2_score > 0:
            self.attributeIndex = lossList.index(best_r2_score)
            self.model = modelList[self.attributeIndex]
            return True
        else:
            self.attributeIndex = -1
            self.yMean = np.average(y, weights=weight)
            return False

    def predict(self, X):
        if self.attributeIndex == -1:
            return self.yMean
        else:
            X_temp = X[:, self.attributeIndex]
            X_temp = X_temp.reshape(-1, 1)
            return self.model.predict(X_temp)
