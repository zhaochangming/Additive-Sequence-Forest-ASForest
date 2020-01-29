from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


def get_RF_result(model, X, y, task):
    result = []

    if task == 'clf':
        prob = []
        for i in range(model.n_estimators):
            prob_temp = model.estimators_[i].predict_proba(X)
            prob.append(prob_temp)
            prob_temp = np.array(prob).mean(axis=0)
            y_pred = prob_temp.argmax(axis=1)
            result.append(accuracy_score(y, y_pred))
    else:
        pred = []
        for i in range(model.n_estimators):
            pred_temp = model.estimators_[i].predict(X)
            pred.append(pred_temp)
            y_pred = np.array(pred).mean(axis=0)
            result.append(mean_squared_error(y, y_pred)**0.5)
    return result
