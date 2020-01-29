from sklearn.metrics import accuracy_score, mean_squared_error


def get_LGBM_result(model, X, y, task):
    result = []
    if task == 'clf':
        for i in range(model.n_estimators):
            prob_temp = model.predict_proba(X, num_iteration=i + 1)
            y_pred = prob_temp.argmax(axis=1)
            result.append(accuracy_score(y, y_pred))
    else:
        for i in range(model.n_estimators):
            y_pred = model.predict(X, num_iteration=i + 1)
            result.append(mean_squared_error(y, y_pred)**0.5)
    return result
