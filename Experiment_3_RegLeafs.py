import os
import pandas as pd
import numpy as np
from src.BoostTree import BT
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from src.BoostForest import BF
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
Num_of_Learners = 100
leaf_nodes = [2, 4, 8, 16, 32, 64, 128, 256]
tree_depth = [1, 2, 3, 4, 5, 6, 7, 8]
N_JOBS = 50
DATA = [
    "CST", "ConcreteFlow", "autoMPG", "RealEstate", "NO2", "PM10", "BH", "CPS", "CCS", "ASN", "ADS",
    "WineWhite", "AQ", "CCPP", "EGSS"
]
save_folder = os.path.join("output3")
params_XGBoost = {
    'min_child_weight': [0.5, 1.0, 3.0],
    'gamma': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
}
params_LGBM = {
    'min_child_weight': [0.5, 1.0, 3.0],
    'min_split_gain': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
}
if __name__ == "__main__":
    print('Experiment 2 Regression')
    kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)
    for d in range(len(DATA)):
        data_name = DATA[d]
        data_path = os.path.join("data", data_name + ".csv")
        data = pd.read_csv(data_path)
        y = data['label'].values
        X = data.drop('label', axis=1).values
        print('load data:', data_name)
        print('X :', X.shape, '|label :', y.shape)

        BoostForest_Loss = []
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
            # RandomForest
            RF_loss = []
            for i in leaf_nodes:
                reg = GridSearchCV(
                    RandomForestRegressor(n_estimators=Num_of_Learners, max_leaf_nodes=i),
                    cv=5,
                    param_grid={"min_samples_leaf": range(5, 16)},
                    iid=True,
                    n_jobs=N_JOBS)
                reg.fit(train_X, train_y)
                test_y_pred = reg.best_estimator_.predict(test_X)
                loss = mean_squared_error(test_y, test_y_pred)**0.5
                RF_loss.append(loss)
            RF_Loss.append(RF_loss)
            # ERT
            ERT_loss = []
            for i in leaf_nodes:
                reg = GridSearchCV(
                    ExtraTreesRegressor(n_estimators=Num_of_Learners, max_leaf_nodes=i),
                    cv=5,
                    param_grid={"min_samples_leaf": range(5, 16)},
                    iid=True,
                    n_jobs=N_JOBS)
                reg.fit(train_X, train_y)
                test_y_pred = reg.best_estimator_.predict(test_X)
                loss = mean_squared_error(test_y, test_y_pred)**0.5
                ERT_loss.append(loss)
            ERT_Loss.append(ERT_loss)
            # XGBoost
            XGB_loss = []
            for i in tree_depth:
                reg = GridSearchCV(
                    XGBRegressor(n_estimators=Num_of_Learners, max_depth=i),
                    cv=5,
                    param_grid=params_XGBoost,
                    iid=True,
                    n_jobs=N_JOBS,
                    scoring='neg_mean_squared_error')
                reg.fit(train_X, train_y)
                test_y_pred = reg.best_estimator_.predict(test_X)
                loss = mean_squared_error(test_y, test_y_pred)**0.5
                XGB_loss.append(loss)
            XGB_Loss.append(XGB_loss)
            # LightGBM
            LGBM_loss = []
            for i in leaf_nodes:
                reg = GridSearchCV(
                    LGBMRegressor(n_estimators=Num_of_Learners, num_leaves=i),
                    cv=5,
                    param_grid=params_LGBM,
                    iid=True,
                    n_jobs=N_JOBS,
                    scoring='neg_mean_squared_error')
                reg.fit(train_X, train_y)
                test_y_pred = reg.best_estimator_.predict(test_X)
                loss = mean_squared_error(test_y, test_y_pred)**0.5
                LGBM_loss.append(loss)
            LGBM_Loss.append(LGBM_loss)

            # 
            BoostForest_loss = []
            for i in leaf_nodes:
                Tree = BT(max_leafs=i, n_jobs=1, task='reg')
                reg = BF(Tree, Num_of_Learners, n_jobs=N_JOBS)
                reg.fit(train_X, train_y, test_X, test_y)
                BoostForest_loss.append(reg.EnsembleLoss[-1])
            BoostForest_Loss.append(BoostForest_loss)


        # Save
        Result = []
        Result.append(RF_Loss)
        Result.append(ERT_Loss)
        Result.append(LGBM_Loss)
        Result.append(BoostForest_Loss)
        Result = np.array(Result)
        save_path = save_folder + '/RMSE_' + data_name
        np.save(save_path, Result)
        # Save XGBoost
        Result = np.array(XGB_Loss)
        save_path = save_folder + '/RMSE_XGB_' + data_name
        np.save(save_path, Result)
