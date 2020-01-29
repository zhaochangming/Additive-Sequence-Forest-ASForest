import os
import numpy as np
from src.ASTree import ASTree
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from src.ASForest import ASForest
from src.Get_RF_Result import get_RF_result
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from src.Get_XGB_Result import get_XGB_result
from src.Get_LGBM_Result import get_LGBM_result
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
Num_of_Learners = 100
N_JOBS = 20
max_leafs = None
save_folder = os.path.join("output2")
DATA = [
    "Sonar",
    "Seeds",
    "Bankruptcy",
    "Column2",
    "Column3",
    "Musk1",
    "BreastCancerDiagnosis",
    "ILPD",
    "bloodDonation",
    "PimaIndiansDiabetes",
    "Vehicle",
    "Biodeg",
    "DiabeticRetinopathyDebrecen",
    "Banknote",
    "WaveForm",
]
params_XGBoost = {
    'min_child_weight': [0.5, 1.0, 3.0],
    'gamma': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'max_depth': [4, 5, 6],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
}
params_LGBM = {
    'min_child_weight': [0.5, 1.0, 3.0],
    'min_split_gain': [0.25, 0.5, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'num_leaves': [16, 32, 64],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0.1, 0.5, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],
}
if __name__ == "__main__":
    ##
    print('Experiment 3 Classification')
    kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=0)
    ##
    for d in range(len(DATA)):
        data_name = DATA[d]
        data_path = os.path.join("data", data_name + ".csv")
        data = pd.read_csv(data_path)
        y = data['label'].values
        X = data.drop('label', axis=1).values
        print('load data:', data_name)
        print('X :', X.shape, '|label :', y.shape)
        _, y = np.unique(y, return_inverse=True)
        #
        ASForest_RI_ACC = []
        ASForest_RC_ACC = []
        RF_ACC = []
        ERT_ACC = []
        XGB_ACC = []
        LGBM_ACC = []
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
            # RandomForest
            clf = GridSearchCV(
                RandomForestClassifier(n_estimators=Num_of_Learners, max_leaf_nodes=max_leafs),
                cv=5,
                param_grid={"min_samples_leaf": range(5, 16)},
                iid=True,
                n_jobs=N_JOBS)
            clf.fit(train_X, train_y)
            clf_best = clf.best_estimator_
            RF_ACC.append(get_RF_result(clf_best, test_X, test_y, 'clf'))

            # ExtraTrees
            clf = GridSearchCV(
                ExtraTreesClassifier(n_estimators=Num_of_Learners, max_leaf_nodes=max_leafs),
                cv=5,
                param_grid={"min_samples_leaf": range(5, 16)},
                iid=True,
                n_jobs=N_JOBS)
            clf.fit(train_X, train_y)
            clf_best = clf.best_estimator_
            ERT_ACC.append(get_RF_result(clf_best, test_X, test_y, 'clf'))

            # XGBoost
            clf = GridSearchCV(
                XGBClassifier(n_estimators=Num_of_Learners),
                cv=5,
                iid=True,
                param_grid=params_XGBoost,
                n_jobs=N_JOBS)
            clf.fit(train_X, train_y)
            XGB_ACC.append(get_XGB_result(clf.best_estimator_, test_X, test_y, 'clf'))

            # LightGBM
            clf = GridSearchCV(
                LGBMClassifier(n_estimators=Num_of_Learners),
                cv=5,
                iid=True,
                param_grid=params_LGBM,
                n_jobs=N_JOBS)
            clf.fit(train_X, train_y)
            LGBM_ACC.append(get_LGBM_result(clf.best_estimator_, test_X, test_y, 'clf'))

            # ASForest_RI
            ASTree_RI = ASTree(max_patch=max_leafs, n_jobs=1, task='clf', RC='F')
            ASForest_RI = ASForest(ASTree_RI, Num_of_Learners, n_jobs=N_JOBS)
            ASForest_RI.fit(train_X, train_y, test_X, test_y)
            ASForest_RI_ACC.append(ASForest_RI.EnsembleACC)
            # ASForest_RC
            ASTree_RC = ASTree(max_patch=max_leafs, n_jobs=1, task='clf', RC='T')
            ASForest_RC = ASForest(ASTree_RC, Num_of_Learners, n_jobs=N_JOBS)
            ASForest_RC.fit(train_X, train_y, test_X, test_y)
            ASForest_RC_ACC.append(ASForest_RC.EnsembleACC)
        # Save
        Result = []
        Result.append(RF_ACC)
        Result.append(ERT_ACC)
        Result.append(XGB_ACC)
        Result.append(LGBM_ACC)
        Result.append(ASForest_RI_ACC)
        Result.append(ASForest_RC_ACC)
        Result = np.array(Result)
        save_path = save_folder + '/ACC_' + data_name
        np.save(save_path, Result)
