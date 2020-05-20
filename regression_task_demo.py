'''
Demo of BoostForest in regression task
@author: ChangmingZhao,Time:2020/05/20
@e-mail: cmzhao@hust.edu.cn
'''
import os
import numpy as np
from src.BoostTree import BT
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from src.BoostForest import BF
import pandas as pd

Num_of_Learners = 50
N_JOBS = 1
max_leafs = None

if __name__ == "__main__":
    ##
    data_name = "CST"
    data_path = os.path.join("data", data_name + ".csv")
    data = pd.read_csv(data_path)
    y = data['label'].values
    X = data.drop('label', axis=1).values
    print('load data:', data_name)
    print('X :', X.shape, '|label :', y.shape)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)
    ##
    min_max_scaler = preprocessing.MinMaxScaler()
    train_X = min_max_scaler.fit_transform(train_X)
    test_X = min_max_scaler.transform(test_X)
    # center y
    standard_sacler = preprocessing.StandardScaler()
    train_y = standard_sacler.fit_transform(train_y.reshape(-1, 1))
    train_y = train_y.reshape(-1, )
    test_y = standard_sacler.transform(test_y.reshape(-1, 1))
    test_y = test_y.reshape(-1, )
    ##
    single_tree = BT(max_leafs=max_leafs, n_jobs=1, task='reg')
    clf = BF(single_tree, Num_of_Learners, n_jobs=N_JOBS)
    clf.fit(train_X, train_y, test_X, test_y)
    print('Ensemble RMSE on testing samples:', clf.EnsembleLoss)
