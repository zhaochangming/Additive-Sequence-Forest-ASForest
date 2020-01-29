import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
Num_of_Learners = [80, 100, 120]
DATA = [
    "CST", "ConcreteFlow", "autoMPG", "RealEstate", "NO2", "PM10", "BH", "CPS", "CCS", "ASN", "ADS",
    "WineWhite", "AQ", "CCPP", "EGSS"
]
N_JOBS = 5
save_folder = os.path.join("output1")
params_XGBoost = {
    'min_child_weight': [5, 10, 15],
    'gamma': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': Num_of_Learners,
}
params_LGBM = {
    'min_child_samples': [5, 10, 15],
    'min_split_gain': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'num_leaves': [16, 32, 64],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': Num_of_Learners,
}
if __name__ == "__main__":
    kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)
    for d in range(len(DATA)):
        data_name = DATA[d]
        data_path = os.path.join("data", data_name + ".csv")
        data = pd.read_csv(data_path)
        y = data['label'].values
        X = data.drop('label', axis=1).values
        print('load data:', data_name)
        print('X :', X.shape, '|label :', y.shape)

        RF_Loss = []
        ERT_Loss = []
        XGB_Loss = []
        LGBM_Loss = []
        Kfold = 0
        for train_index, test_index in kf.split(X):
            Kfold += 1
            print('Dataset: ', d + 1, 'Kfold: ', Kfold)
            # MinMaxScaler
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            min_max_scaler = preprocessing.MinMaxScaler()
            train_X = min_max_scaler.fit_transform(train_X)
            test_X = min_max_scaler.transform(test_X)
            # center y
            standard_sacler = preprocessing.StandardScaler()
            train_y = standard_sacler.fit_transform(train_y.reshape(-1, 1))
            train_y = train_y.reshape(-1, )
            test_y = standard_sacler.transform(test_y.reshape(-1, 1))
            test_y = test_y.reshape(-1, )
            # RF
            reg = GridSearchCV(
                RandomForestRegressor(),
                cv=5,
                iid=True,
                param_grid={
                    "min_samples_leaf": range(5, 16),
                    "n_estimators": Num_of_Learners
                },
                n_jobs=N_JOBS)
            reg.fit(train_X, train_y)
            test_y_pred = reg.best_estimator_.predict(test_X)
            loss = mean_squared_error(test_y, test_y_pred)**0.5
            RF_Loss.append(loss)
            # ERT
            reg = GridSearchCV(
                ExtraTreesRegressor(),
                cv=5,
                iid=True,
                param_grid={
                    "min_samples_leaf": range(5, 16),
                    "n_estimators": Num_of_Learners
                },
                n_jobs=N_JOBS)
            reg.fit(train_X, train_y)
            test_y_pred = reg.best_estimator_.predict(test_X)
            loss = mean_squared_error(test_y, test_y_pred)**0.5
            ERT_Loss.append(loss)
            # LightGBM
            reg = GridSearchCV(
                LGBMRegressor(),
                cv=5,
                iid=True,
                param_grid=params_LGBM,
                n_jobs=N_JOBS,
                scoring='neg_mean_squared_error')
            reg.fit(train_X, train_y)
            test_y_pred = reg.best_estimator_.predict(test_X)
            loss = mean_squared_error(test_y, test_y_pred)**0.5
            LGBM_Loss.append(loss)
            # XGBoost
            reg = GridSearchCV(
                XGBRegressor(),
                cv=5,
                param_grid=params_XGBoost,
                iid=True,
                n_jobs=N_JOBS,
                scoring='neg_mean_squared_error')
            reg.fit(train_X, train_y)
            test_y_pred = reg.best_estimator_.predict(test_X)
            loss = mean_squared_error(test_y, test_y_pred)**0.5
            XGB_Loss.append(loss)

        # Save
        Result = []
        Result.append(RF_Loss)
        Result.append(ERT_Loss)
        Result.append(LGBM_Loss)
        Result.append(XGB_Loss)
        Result = np.array(Result)
        save_path = save_folder + '/RMSE_' + data_name
        np.save(save_path, Result)
