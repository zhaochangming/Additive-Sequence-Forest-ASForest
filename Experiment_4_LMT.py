import os
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
import pandas as pd
from src.LogisticModelTree import LMT
from src.ASForest_LMT import ASForest_LMT
TASK = 'clf'
Num_of_Learners = 100
N_JOBS = 50
save_folder = os.path.join("output4")
DATA = [
    "Sonar",
    "Seeds",
    "Bankruptcy",
    "Column2",
    "Column3",
    "Musk1",
    "ClimateModel",
    "BreastCancerDiagnosis",
    "ILPD",
    "bloodDonation",
    "PimaIndiansDiabetes",
    "Vehicle",
    "Biodeg",
    "DiabeticRetinopathyDebrecen",
    "Banknote",
    "Steel",
    "WaveForm",
]
if __name__ == "__main__":
    ##
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
        ASForestLMT_RI_ACC = []
        ASForestLMT_RC_ACC = []
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
            # LMT_RI
            clf = LMT(RC='F')
            ASForestLMT_RI = ASForest_LMT(clf, Num_of_Learners, n_jobs=N_JOBS)
            ASForestLMT_RI.fit(train_X, train_y, test_X, test_y)
            ASForestLMT_RI_ACC.append(ASForestLMT_RI.EnsembleACC[-1])
            # LMT_RT
            clf = LMT(RC='T')
            ASForestLMT_RC = ASForest_LMT(clf, Num_of_Learners, n_jobs=N_JOBS)
            ASForestLMT_RC.fit(train_X, train_y, test_X, test_y)
            ASForestLMT_RC_ACC.append(ASForestLMT_RC.EnsembleACC[-1])

        # Save
        Result = []
        Result.append(ASForestLMT_RI_ACC)
        Result.append(ASForestLMT_RC_ACC)
        Result = np.array(Result)
        save_path = save_folder + '/ACC_' + data_name
        np.save(save_path, Result)
