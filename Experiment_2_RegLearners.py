import os
import pandas as pd
import numpy as np
from src.ASTree import ASTree
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.Get_XGB_Result import get_XGB_result
from src.Get_LGBM_Result import get_LGBM_result
from src.ASForest import ASForest
from sklearn.model_selection import GridSearchCV
from src.Get_RF_Result import get_RF_result
Num_of_Learners = 100
N_JOBS = 50
max_leafs = None
DATA = [
    "CST", "ConcreteFlow", "autoMPG", "RealEstate", "NO2", "PM10", "BH", "CPS", "CCS", "ASN", "ADS",
    "WineWhite", "AQ", "CCPP", "EGSS"
]
save_folder = os.path.join("output2")
params_XGBoost = {
    'min_child_weight': [5, 10, 15],
    'gamma': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
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
}

if __name__ == "__main__":
    print('Experiment 3 Regression')
    kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)
    for d in range(len(DATA)):
        data_name = DATA[d]
        data_path = os.path.join("data", data_name + ".csv")
        data = pd.read_csv(data_path)
        y = data['label'].values
        X = data.drop('label', axis=1).values
        print('load data:', data_name)
        print('X :', X.shape, '|label :', y.shape)

        ASForest_RI_Loss = []
        ASForest_RC_Loss = []
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
            reg = GridSearchCV(
                RandomForestRegressor(n_estimators=Num_of_Learners, max_leaf_nodes=max_leafs),
                cv=5,
                param_grid={"min_samples_leaf": range(5, 16)},
                iid=True,
                n_jobs=N_JOBS)
            reg.fit(train_X, train_y)
            reg_best = reg.best_estimator_
            RF_Loss.append(get_RF_result(reg_best, test_X, test_y, 'reg'))

            # ERT
            reg = GridSearchCV(
                ExtraTreesRegressor(n_estimators=Num_of_Learners, max_leaf_nodes=max_leafs),
                cv=5,
                param_grid={"min_samples_leaf": range(5, 16)},
                iid=True,
                n_jobs=N_JOBS)
            reg.fit(train_X, train_y)
            ERT_Loss.append(get_RF_result(reg.best_estimator_, test_X, test_y, 'reg'))

            # XGBoost
            reg = GridSearchCV(
                XGBRegressor(n_estimators=Num_of_Learners),
                cv=5,
                param_grid=params_XGBoost,
                iid=True,
                n_jobs=N_JOBS)
            reg.fit(train_X, train_y)
            XGB_Loss.append(get_XGB_result(reg.best_estimator_, test_X, test_y, 'reg'))

            # LightGBM
            reg = GridSearchCV(
                LGBMRegressor(n_estimators=Num_of_Learners),
                cv=5,
                param_grid=params_LGBM,
                iid=True,
                n_jobs=N_JOBS,
                scoring='neg_mean_squared_error')
            reg.fit(train_X, train_y)
            LGBM_Loss.append(get_LGBM_result(reg.best_estimator_, test_X, test_y, 'reg'))
            # ASForest_RI
            ASTree_RI = ASTree(max_patch=max_leafs, n_jobs=1, task='reg', RC='F')
            ASForest_RI = ASForest(ASTree_RI, Num_of_Learners, n_jobs=N_JOBS)
            ASForest_RI.fit(train_X, train_y, test_X, test_y)
            ASForest_RI_Loss.append(ASForest_RI.EnsembleLoss)
            # ASForest_RC
            ASTree_RC = ASTree(max_patch=max_leafs, n_jobs=1, task='reg', RC='T')
            ASForest_RC = ASForest(ASTree_RC, Num_of_Learners, n_jobs=N_JOBS)
            ASForest_RC.fit(train_X, train_y, test_X, test_y)
            ASForest_RC_Loss.append(ASForest_RC.EnsembleACC)
        # Save
        Result = []
        Result.append(RF_Loss)
        Result.append(ERT_Loss)
        Result.append(XGB_Loss)
        Result.append(LGBM_Loss)
        Result.append(ASForest_RI_Loss)
        Result.append(ASForest_RC_Loss)
        Result = np.array(Result)
        save_path = save_folder + '/RMSE_' + data_name
        np.save(save_path, Result)
