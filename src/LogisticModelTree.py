import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from scipy.special import softmax, expit
import random
from math import ceil, sqrt
from models.SingleLinearRegression import SingleLinearRegression
from sklearn.model_selection import KFold
_MACHINE_EPSILON = np.finfo(np.float64).eps
max_response = 4
NUM_SPLIT = 100
sample_leaf_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
MaxIterations = 200
HeuristicStop = 20


class LMT(object):
    def __init__(self, RC='F'):
        self.tree = None
        self.train_ACC = []
        self.train_X = None
        self.train_y = None
        self.leaf_num = 1
        self.verbose = False
        self.n_classes = None
        self.feature_demension = None
        self.split_feature_demension = None
        self.has_test = None
        self.train_label = None
        self.RC = RC

    # ====================
    # Fit
    # ====================
    def fit(self, train_X, train_y, test_X=None, test_y=None, verbose=False):
        # Settings
        self.verbose = verbose
        self.train_X = train_X
        self.train_label = train_y
        self.feature_demension = self.train_X.shape[1]
        self.split_feature_demension = ceil(sqrt(self.feature_demension))
        self.feature_candidate = range(self.feature_demension)
        # training options
        self.n_classes = np.unique(train_y).shape[0]
        if self.n_classes == 2:
            self.train_y = train_y
        else:
            self.train_y = np.eye(self.n_classes)[train_y]

        # testing options
        self.has_test = (test_X is not None) and (test_y is not None)
        if self.has_test:
            self.test_X = test_X
            self.test_label = test_y
            self.test_ACC = []
            if self.n_classes == 2:
                self.test_y = test_y
            else:
                self.test_y = np.eye(self.n_classes)[test_y]

        # Construct tree
        self._build_tree()

    # ====================
    # Build Tree
    # ====================

    def _build_tree(self):
        X, y = self.train_X, self.train_y
        if self.n_classes == 2:
            output = np.zeros(len(X), dtype=np.float64)
        else:
            output = np.zeros((len(X), self.n_classes), dtype=np.float64)
        # fit root model
        data = (X, y, output)
        model_root, data = self.fit_model(data)
        #
        self.container = 0  # mutatable container
        self.tree = self._create_node(data, 0, model_root)
        # split and traverse root node
        self._split_traverse_node(self.tree)

    # ====================
    # Recursively split node + traverse node until a terminal node is reached
    # ====================
    def _split_traverse_node(self, node):
        """
            main loop
        """
        # Perform split and collect result
        result = self._splitter(node)
        # Return terminal node if split is not advised
        if not result["did_split"]:
            return

        # Update node information based on splitting result
        node["j_feature"] = result["j_feature"]
        node["threshold"] = result["threshold"]
        del node["data"]  # delete node stored data

        # Extract splitting results
        data_left, data_right = result["data"]
        # Fit Model
        model_left, data_left = self.fit_model(data_left)
        model_right, data_right = self.fit_model(data_right)
        # Create children nodes
        node["children"]["left"] = self._create_node(data_left, node["depth"] + 1, model_left)
        node["children"]["right"] = self._create_node(data_right, node["depth"] + 1, model_right)
        # Split Node
        self.leaf_num += 1
        # self.train_acc_test()
        self._split_traverse_node(node["children"]["left"])
        self._split_traverse_node(node["children"]["right"])

    def _splitter(self, node):
        """
        Split the node and collect result
        """
        # Extract data
        X, y, output = node["data"]
        N = node["n_samples"]
        # Find feature splits that might improve loss
        did_split = False
        gini_best = node["gini"]
        data_best = None
        j_feature = None
        threshold = None
        X_split = X[:, node["random_feature_index"]]
        if self.RC == 'T':
            X_split = X_split @ node["random_filter"]
        data_split = (X_split, y, output)
        feature_threshold = get_feature_threshold(X_split, N)
        if feature_threshold:
            split_result_list = []
            for f_t in feature_threshold:
                split_result_list.append(self._split_gini(data_split, f_t))
            best_gini_split = min(split_result_list)
            best_gini_index = split_result_list.index(best_gini_split)
            j_feature, threshold = feature_threshold[best_gini_index]
            if best_gini_split < gini_best:
                did_split = True
                idx_left, idx_right = split_idx(j_feature, threshold, X_split)
                data_left, data_right = split_data(node["data"], idx_left, idx_right)
                data_best = [data_left, data_right]
        # Return the best result
        result = {
            "did_split": did_split,
            "data": data_best,
            "j_feature": j_feature,
            "threshold": threshold
        }

        return result

    # ====================
    # Create Node
    # ====================
    def _create_node(self, data, depth, model_node=None):
        #
        X, y, output = data
        random_feature_index = random.sample(self.feature_candidate, self.split_feature_demension)
        #
        if self.RC == 'T':
            random_filter = np.random.rand(self.split_feature_demension,
                                           self.split_feature_demension)
        else:
            random_filter = None
        gini = self.getGini(y, output)
        #
        node = {
            "name": "node",
            "index": self.container,
            "gini": gini,
            "model": model_node,
            "data": data,
            "n_samples": len(X),
            "j_feature": None,
            "threshold": None,
            "children": {
                "left": None,
                "right": None
            },
            "depth": depth,
            "random_feature_index": random_feature_index,
            "random_filter": random_filter,
        }
        self.container += 1
        return node

    def predict_x(self, node, x, y_pred_x=None):
        no_children = node["children"]["left"] is None and node["children"]["right"] is None
        isRoot = (y_pred_x is None)
        if isRoot:
            if self.n_classes == 2:
                y_pred_x = np.zeros((1, 1), dtype=np.float64)
            else:
                y_pred_x = np.zeros((1, self.n_classes), dtype=np.float64)

        if node["model"]:
            for model_temp in node["model"]:
                new_scores = self.predict_score(model_temp, x.reshape(1, -1))
                y_pred_x += new_scores

        if no_children:
            return y_pred_x
        else:
            x_split = x[node["random_feature_index"]]
            if self.RC == 'T':
                x_split = x_split.reshape(1, -1) @ node["random_filter"]
                x_split = x_split[0]
            if x_split[node["j_feature"]] <= node["threshold"]:  # x[j] <= threshold
                return self.predict_x(node["children"]["left"], x, y_pred_x)
            else:  # x[j] > threshold
                return self.predict_x(node["children"]["right"], x, y_pred_x)

    def predict_prob_output(self, X):
        assert self.tree is not None
        y_pred = np.array([self.predict_x(self.tree, x) for x in X])
        y_pred = y_pred.reshape(len(X), -1)
        prob = self.output2prob(y_pred)
        return prob, y_pred

    def output2prob(self, output):
        if self.n_classes == 2:
            prob = expit(output)
        else:
            prob = softmax(output, axis=1)
        return prob

    def prob2pred_label(self, prob):
        if self.n_classes == 2:
            prob_temp = np.c_[1 - prob, prob]
            y_pred = prob_temp.argmax(axis=1)
        else:
            y_pred = prob.argmax(axis=1)
        return y_pred

    def predict_score(self, node_model, X):
        if self.n_classes == 2:
            new_scores = node_model.predict(X)
        else:
            new_scores = [e.predict(X) for e in node_model]
            new_scores = np.asarray(new_scores).T
            new_scores -= new_scores.mean(keepdims=True)
            new_scores *= (self.n_classes - 1) / self.n_classes
        return new_scores

    def getGini(self, y, output):
        prob = self.output2prob(output)
        gini = 0
        if self.n_classes == 2:
            weight, z = _weight_and_response(y, prob)
            gini_temp = entropy(weight, z)
            gini += gini_temp
        else:
            for j in range(self.n_classes):
                weight, z = _weight_and_response(y[:, j], prob[:, j])
                gini_temp = entropy(weight, z)
                gini += gini_temp
        return gini

    def train_acc_test(self):
        _, output = self.predict_prob_output(self.train_X)
        data = (None, self.train_y, output)
        print(self.predict_acc(data))

    def fit_model(self, data):
        (X, y, output) = data
        completedIterations = MaxIterations
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        numFoldsACCs = []
        for train_index, val_index in kf.split(X):
            train_X, train_y = X[train_index], y[train_index]
            val_X, val_y = X[val_index], y[val_index]
            train_output, val_output = output[train_index], output[val_index]
            train_data = (train_X, train_y, train_output)
            val_data = (val_X, val_y, val_output)
            val_accs, iterations = self.performBoosting(train_data, completedIterations, val_data)
            if iterations < completedIterations:
                completedIterations = iterations
            numFoldsACCs.append(val_accs)
        bestIteration = getBestIteration(numFoldsACCs, completedIterations)
        if bestIteration > 0:
            model, data = self.performBoosting(data, bestIteration)
            return model, data
        else:
            return None, data

    def performBoosting(self, train_data, completedIterations, val_data=None):
        model = []
        noMax = 0
        if val_data is not None:
            val_accs = []
            lastMax = self.predict_acc(val_data)
            val_accs.append(lastMax)
            for i in range(completedIterations):
                foundAttribute, model_temp, train_data, val_data = self._fit_model_iteration(
                    train_data, val_data)
                if foundAttribute:
                    model.append(model_temp)
                    val_accs.append(self.predict_acc(val_data))
                else:
                    break
                if noMax > HeuristicStop:
                    break
                if val_accs[-1] > lastMax:
                    lastMax = val_accs[-1]
                    noMax = 0
                else:
                    noMax += 1
            return val_accs, len(val_accs) - 1
        else:
            for i in range(completedIterations):
                foundAttribute, model_temp, train_data, _ = self._fit_model_iteration(train_data)
                if foundAttribute:
                    model.append(model_temp)
                else:
                    break
            return model, train_data

    def predict_acc(self, data):
        (X, y, output) = data
        y_prob = self.output2prob(output)
        y_pred = self.prob2pred_label(y_prob)
        if self.n_classes > 2:
            y = self.prob2pred_label(y)
        return accuracy_score(y, y_pred)

    def _fit_model_iteration(self, data, val_data=None):
        (X, y, output) = data
        prob = self.output2prob(output)
        model = SingleLinearRegression()
        if self.n_classes == 2:
            weight, z = _weight_and_response(y, prob)
            X_train, z_train, weight_train = filter_quantile(X, z, weight, trim_quantile=0.05)
            new_estimators = deepcopy(model)  # must deepcopy the model!
            foundAttribute = new_estimators.fit(X_train, z_train, weight_train)
            if foundAttribute == False:
                return foundAttribute, None, data, val_data
        else:
            new_estimators = []
            for j in range(self.n_classes):
                # weight
                weight, z = _weight_and_response(y[:, j], prob[:, j])
                # filter
                X_train, z_train, weight_train = filter_quantile(X, z, weight, trim_quantile=0.05)
                model_copy = deepcopy(model)  # must deepcopy the model!
                foundAttribute = model_copy.fit(X_train, z_train, weight_train)
                if foundAttribute == False:
                    return foundAttribute, None, data, val_data
                new_estimators.append(model_copy)
        new_scores = self.predict_score(new_estimators, X)
        y_pred = new_scores + output
        if val_data is not None:
            (val_X, val_y, val_output) = val_data
            new_scores = self.predict_score(new_estimators, val_X)
            val_y_pred = new_scores + val_output
            return foundAttribute, new_estimators, (X, y, y_pred), (val_X, val_y, val_y_pred)
        else:
            return foundAttribute, new_estimators, (X, y, y_pred), None

    def _split_gini(self, data, f_t):
        # Split data based on threshold
        j_feature, threshold = f_t
        X_split, y, output = data
        idx_left, idx_right = split_idx(j_feature, threshold, X_split)
        data_left, data_right = split_data(data, idx_left, idx_right)
        _, y_left, output_left = data_left
        _, y_right, output_right = data_right
        gini_left = self.getGini(y_left, output_left)
        gini_right = self.getGini(y_right, output_right)
        return gini_left + gini_right


def filter_quantile(X, z, sample_weight, trim_quantile):
    threshold = np.quantile(sample_weight, trim_quantile, interpolation="lower")
    mask = (sample_weight >= threshold)
    X_train = X[mask]
    z_train = z[mask]
    sample_weight = sample_weight[mask]
    return X_train, z_train, sample_weight


def getBestIteration(ACCs, completedIterations):
    best_ACC = 0
    best_index = 0
    for i in range(completedIterations + 1):
        ACC_temp = ACCs[0][i] + ACCs[1][i] + ACCs[2][i] + ACCs[3][i] + ACCs[4][i]
        if ACC_temp > best_ACC:
            best_index = i
            best_ACC = ACC_temp
    return best_index


def _weight_and_response(y, prob):
    sample_weight = prob * (1. - prob)
    sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)
    with np.errstate(divide="ignore", over="ignore"):
        z = np.where(y, 1. / prob, -1. / (1. - prob))
    z = np.clip(z, a_min=-max_response, a_max=max_response)
    return sample_weight, z


def entropy(weight, z):
    m = np.average(z, weights=weight)
    e = 0
    numInstances = len(z)
    for i in range(numInstances):
        e += weight[i] * pow(z[i] - m, 2)
    return e


def split_idx(j_feature, threshold, X_split):
    idx_left = np.where(X_split[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X_split)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X_split)
    return idx_left, idx_right


def get_feature_threshold(X, N):
    # leaf samples
    min_samples_left, min_samples_right = random.sample(sample_leaf_list, 2)
    # feature_threshold
    feature_threshold = []
    if N >= (min_samples_left + min_samples_right):
        for j_feature in range(X.shape[1]):
            threshold_search = []
            for i in range(N):
                threshold_search.append(X[i, j_feature])  # round
            threshold_search = list(set(threshold_search))

            if len(threshold_search) > NUM_SPLIT:
                value_min, value_max = min(threshold_search), max(threshold_search)
                threshold_search = np.linspace(value_min, value_max, num=NUM_SPLIT)
                threshold_search = list(np.around(threshold_search, decimals=3))
                threshold_search = list(set(threshold_search))

            for threshold in threshold_search:
                idx_left, idx_right = split_idx(j_feature, threshold, X)
                if (len(idx_left) >= min_samples_left) and (len(idx_right) >= min_samples_right):
                    feature_threshold.append((j_feature, threshold))
        # random threshold
        N_threshold = len(feature_threshold)
        idx = random.sample(range(N_threshold), ceil(sqrt(N_threshold)))
        feature_threshold_random = []
        for i in idx:
            feature_threshold_random.append(feature_threshold[i])
        feature_threshold = feature_threshold_random
    return feature_threshold


def split_data(data, idx_left, idx_right):
    X, y, output = data
    data_left = (X[idx_left], y[idx_left], output[idx_left])
    data_right = (X[idx_right], y[idx_right], output[idx_right])
    return data_left, data_right
