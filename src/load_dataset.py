import scipy.io
import numpy as np
import h5py


def load_dataset_h5py(path):
    dataset = h5py.File(path)
    data = np.float64(dataset['data']).T
    X = data[:, 0:-1]
    X.dtype = "float"
    y = data[:, -1]
    print('load data:', path)
    print('X :', X.shape, '|label :', y.shape)
    return X, y


def load_dataset_scipy(path):
    dataset = scipy.io.loadmat(path)
    data = np.float64(dataset['data'])
    X = data[:, 0:-1]
    X.dtype = "float"
    y = data[:, -1]
    print('load data:', path)
    print('X :', X.shape, '|label :', y.shape)
    return X, y


def load_dataset(path):
    try:
        X, y = load_dataset_h5py(path)
    except Exception as e:
        X, y = load_dataset_scipy(path)
    return X, y
