import os
import numpy as np
from src.BoostTree import BT
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from src.BoostForest import BF
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
Num_of_Learners = 100
N_JOBS = 30
leaf_nodes = [2, 3, 4, 6, 8, 10, 13, 16, 20, 26, 32]
tree_depth = [1, 2, 3, 4, 5]
save_folder = os.path.join("output3")
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
    ##
    print('Experiment 2 Classification')
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
        BoostForest_ACC = []
        RF_ACC = []
        ERT_ACC = []
        LGBM_ACC = []
        XGB_ACC = []
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
            RF_acc = []
            for i in leaf_nodes:
                clf = GridSearchCV(
                    RandomForestClassifier(n_estimators=Num_of_Learners, max_leaf_nodes=i),
                    cv=5,
                    param_grid={"min_samples_leaf": range(5, 16)},
                    iid=True,
                    n_jobs=N_JOBS)
                clf.fit(train_X, train_y)
                RF_acc.append(clf.best_estimator_.score(test_X, test_y))
            RF_ACC.append(RF_acc)

            # ExtraTrees
            ERT_acc = []
            for i in leaf_nodes:
                clf = GridSearchCV(
                    ExtraTreesClassifier(n_estimators=Num_of_Learners, max_leaf_nodes=i),
                    cv=5,
                    param_grid={"min_samples_leaf": range(5, 16)},
                    iid=True,
                    n_jobs=N_JOBS)
                clf.fit(train_X, train_y)
                ERT_acc.append(clf.best_estimator_.score(test_X, test_y))
            ERT_ACC.append(ERT_acc)

            # XGBoost
            XGB_acc = []
            for i in tree_depth:
                clf = GridSearchCV(
                    XGBClassifier(n_estimators=Num_of_Learners, max_depth=i),
                    cv=5,
                    iid=True,
                    param_grid=params_XGBoost,
                    n_jobs=N_JOBS)
                clf.fit(train_X, train_y)
                XGB_acc.append(clf.best_estimator_.score(test_X, test_y))
            XGB_ACC.append(XGB_acc)

            # LightGBM
            LGBM_acc = []
            for i in leaf_nodes:
                clf = GridSearchCV(
                    LGBMClassifier(n_estimators=Num_of_Learners, num_leaves=i),
                    cv=5,
                    iid=True,
                    param_grid=params_LGBM,
                    n_jobs=N_JOBS)
                clf.fit(train_X, train_y)
                LGBM_acc.append(clf.best_estimator_.score(test_X, test_y))
            LGBM_ACC.append(LGBM_acc)

            BoostForest_acc = []
            for i in leaf_nodes:
                Tree = BT(max_leafs=i, n_jobs=1, task='clf')
                clf = BF(Tree, Num_of_Learners, n_jobs=N_JOBS)
                clf.fit(train_X, train_y, test_X, test_y)
                BoostForest_acc.append(clf.EnsembleACC[-1])
            BoostForest_ACC.append(BoostForest_acc)

        # Save
        Result = []
        Result.append(RF_ACC)
        Result.append(ERT_ACC)
        Result.append(LGBM_ACC)
        Result.append(BoostForest_ACC)
        Result = np.array(Result)
        save_path = save_folder + '/ACC_' + data_name
        np.save(save_path, Result)
        # save XGB
        Result = np.array(XGB_ACC)
        save_path = save_folder + '/ACC_XGB_' + data_name
        np.save(save_path, Result)
