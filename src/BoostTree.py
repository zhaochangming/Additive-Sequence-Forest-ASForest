"""

BoostTree

"""

import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from scipy.special import softmax, expit
import random
from math import ceil, sqrt
from models.weight_ridge import weight_ridge
from joblib import Parallel, delayed

_MACHINE_EPSILON = np.finfo(np.float64).eps
max_response = 4
NUM_SPLIT = 100
sample_leaf_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class BT(object):
    def __init__(self, max_leafs=5, n_jobs=1, task='clf', RC='F'):
        self.max_leafs = max_leafs
        self.tree = None
        self.train_Loss = []
        self.train_X = None
        self.train_y = None
        self.leaf_num = 1
        self.verbose = False
        self.n_classes = None
        self.feature_demension = None
        self.split_feature_demension = None
        self.categorical_feature_index = []
        self.n_jobs = n_jobs
        # task:['reg','clf']
        self.task = task
        self.has_test = None
        self.parallel_data = None
        self.RC = RC
        if task == 'clf':
            self.L2_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
        else:
            self.L2_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
            # self.L2_list = [1, 0.1, 0.01, 0.001]

    def get_params(self, deep=True):
        return {
            "max_leafs": self.max_leafs,
        }

    # ======================
    # Fit
    # ======================
    def fit(self, train_X, train_y, test_X=None, test_y=None, verbose=False):

        # Settings
        self.verbose = verbose
        self.train_X = train_X
        self.train_label = train_y
        self.feature_demension = self.train_X.shape[1]
        self.split_feature_demension = ceil(sqrt(self.feature_demension))

        # training options
        if self.task == 'clf':
            self.n_classes = np.unique(train_y).shape[0]
            self.train_ACC = []
            if self.n_classes == 2:
                self.train_y = train_y
            else:
                self.train_y = np.eye(self.n_classes)[train_y]
        else:
            self.n_classes = 1
            self.train_y = train_y

        # testing options
        self.has_test = (test_X is not None) and (test_y is not None)
        if self.has_test:
            self.test_X = test_X
            self.test_label = test_y
            self.test_Loss = []
            if self.task == 'clf':
                self.test_ACC = []
                if self.n_classes == 2:
                    self.test_y = test_y
                else:
                    self.test_y = np.eye(self.n_classes)[test_y]
            else:
                self.test_y = test_y

        if self.verbose:
            print(" max_leafs={}, \n alpha_list={}, \n sample_leaf_list={}".format(
                self.max_leafs, self.L2_list, sample_leaf_list))
        # categorical_feature_index
        for j_feature in range(self.feature_demension):
            if len(np.unique(self.train_X[:, j_feature])) == 2:
                self.categorical_feature_index.append(j_feature)
        # Construct tree
        self._build_tree()
        del self.train_X, self.train_y, self.train_label

    # ======================
    # Predict Prob
    # ======================
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

    def predict_score(self, node_model, X):
        X_continus = np.delete(X, self.categorical_feature_index, axis=1)
        if (self.task == 'reg') or (self.n_classes == 2):
            new_scores = node_model.predict(X_continus)
        else:
            new_scores = [e.predict(X_continus) for e in node_model]
            new_scores = np.asarray(new_scores).T
            new_scores -= new_scores.mean(keepdims=True)
            new_scores *= (self.n_classes - 1) / self.n_classes
        return new_scores

    def predict_x(self, node, x, y_pred_x=None):
        no_children = node["children"]["left"] is None and node["children"]["right"] is None
        if no_children:
            if node["model"] is None:
                if (self.task == 'reg') or (self.n_classes == 2):
                    y_pred_x = 0
                else:
                    y_pred_x = np.zeros((1, self.n_classes), dtype=np.float64)
                return y_pred_x
            else:
                new_scores = self.predict_score(node["model"], x.reshape(1, -1))
            y_pred_x += new_scores
            return y_pred_x
        else:
            if node["model"] is None:
                if (self.task == 'reg') or (self.n_classes == 2):
                    y_pred_x = 0
                else:
                    y_pred_x = np.zeros((1, self.n_classes), dtype=np.float64)

            else:
                new_scores = self.predict_score(node["model"], x.reshape(1, -1))
                y_pred_x += new_scores
            if self.RC == 'T':
                x_split = x[node["random_feature_index"]]
                x_split = x_split.reshape(1, -1) @ node["random_filter"]
                x_split = x_split[0]
            else:
                x_split = x[node["random_feature_index"]]
            if x_split[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                return self.predict_x(node["children"]["left"], x, y_pred_x)
            else:  # x[j] > threshold
                return self.predict_x(node["children"]["right"], x, y_pred_x)

    # ======================
    # Loss
    # ======================
    def loss(self, y, y_pred):
        if self.task == 'clf':
            if self.n_classes == 2:
                loss = log_loss(y, y_pred, labels=[0, 1])
            else:
                loss = log_loss(y, y_pred, labels=np.eye(self.n_classes))
        else:
            loss = mean_squared_error(y, y_pred)**0.5
        return loss

    def prob2pred_label(self, prob):
        if self.n_classes == 2:
            prob_temp = np.c_[1 - prob, prob]
            y_pred = prob_temp.argmax(axis=1)
        else:
            y_pred = prob.argmax(axis=1)
        return y_pred

    ##
    # predict stepwise
    ##
    def predict_stagewise(self):
        if self.task == 'clf':
            # training acc, training loss
            y_prob, _ = self.predict_prob_output(self.train_X)
            self.train_Loss.append(self.loss(self.train_y, y_prob))
            y_pred = self.prob2pred_label(y_prob)
            self.train_ACC.append(accuracy_score(self.train_label, y_pred))
            # testing acc, testing loss
            if self.has_test:
                y_prob, _ = self.predict_prob_output(self.test_X)
                self.test_Loss.append(self.loss(self.test_y, y_prob))
                y_pred = self.prob2pred_label(y_prob)
                self.test_ACC.append(accuracy_score(self.test_label, y_pred))
        else:
            # training loss
            _, y_pred = self.predict_prob_output(self.train_X)
            self.train_Loss.append(self.loss(self.train_y, y_pred))
            # testing loss
            if self.has_test:
                _, y_pred = self.predict_prob_output(self.test_X)
                self.test_Loss.append(self.loss(self.test_y, y_pred))

    def _build_tree(self):
        X, y = self.train_X, self.train_y
        if (self.task == 'reg') or (self.n_classes == 2):
            output = np.zeros(len(X), dtype=np.float64)
        else:
            output = np.zeros((len(X), self.n_classes), dtype=np.float64)

        container = {"index_node_global": 0}  # mutatable container
        if self.task == 'clf':
            prob = self.output2prob(output)
            loss_root = self.loss(y, prob)
        else:
            loss_root = self.loss(y, output)
        data = (X, y, output)
        self.tree = self._create_node(data, 0, container, loss_root)  # depth 0 root node
        ##
        node_temp = []
        node_temp.append(self.tree)
        loss_temp = []
        loss_temp.append(self.tree["n_samples"] * self.tree["loss"])
        # split and traverse root node
        split_index = 0
        while split_index >= 0:
            node_temp, loss_temp, split_index = self._split_traverse_node(
                node_temp[split_index], container, node_temp, loss_temp)
        # del data
        while True:
            if max(loss_temp) == 0:
                break
            else:
                max_index = loss_temp.index(max(loss_temp))
                loss_temp[node_temp[max_index]["index"]] = 0
                del node_temp[max_index]["data"]

    # Recursively split node + traverse node until a terminal node is reached
    def _split_traverse_node(self, node, container, node_temp, loss_temp):
        """
            main loop
        """
        # Perform split and collect result
        result = self._splitter(node)
        del node["data"]
        loss_temp[node["index"]] = 0
        # Return terminal node if split is not advised
        if not result["did_split"]:

            if self.verbose:
                depth_spacing_str = " ".join([" "] * node["depth"])
                print(" {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(
                    depth_spacing_str, node["index"], node["depth"], node["loss"],
                    node["n_samples"]))
            # #
            if self.max_leafs is None:
                max_leaf_condition = True
            else:
                max_leaf_condition = (self.leaf_num < self.max_leafs)
            if max_leaf_condition:
                # del data in leaf node
                if max(loss_temp) == 0:
                    return node_temp, loss_temp, -1
                split_index = loss_temp.index(max(loss_temp))
                return node_temp, loss_temp, split_index
            return node_temp, loss_temp, -1

        # Update node information based on splitting result
        node["j_feature"] = result["j_feature"]
        node["threshold"] = result["threshold"]

        # Extract splitting results
        data_left, data_right = result["data"]
        model_left, model_right = result["models"]
        loss_left, loss_right = result["loss"]
        N_left, N_right = result["N_left_right"]

        # Report created node to user
        if self.verbose:
            depth_spacing_str = " ".join([" "] * node["depth"])
            print(" {}node {} @ depth {}: loss={:.6f}, j_feature={}, threshold={:.6f}, N=({},{})".
                  format(depth_spacing_str, node["index"], node["depth"], node["loss"],
                         node["j_feature"], node["threshold"], N_left, N_right))
        
        # Create children nodes
        node["children"]["left"] = self._create_node(data_left, node["depth"] + 1, container,
                                                     loss_left, model_left)
        node_temp.append(node["children"]["left"])
        loss_temp.append(node["children"]["left"]["n_samples"] * node["children"]["left"]["loss"])
        node["children"]["right"] = self._create_node(data_right, node["depth"] + 1, container,
                                                      loss_right, model_right)
        node_temp.append(node["children"]["right"])
        loss_temp.append(node["children"]["right"]["n_samples"] * node["children"]["right"]["loss"])

        # self.predict_stagewise()  # debug
        self.leaf_num += 1
        # decide split node
        if max(loss_temp) == 0:
            return node_temp, loss_temp, -1
        split_index = loss_temp.index(max(loss_temp))
        return node_temp, loss_temp, split_index

    def _splitter(self, node):
        """
        Split the node and collect result
        """
        # Extract data
        X, y, output = node["data"]
        data = node["data"]
        N = node["n_samples"]
        # Find feature splits that might improve loss
        did_split = False
        loss_best = node["loss"]
        data_best = None
        models_best = None
        N_left_right = None
        j_feature = None
        threshold = None
        #
        if node["random_feature_index"] is not None:
            if self.RC == 'T':
                X_split = X[:, node["random_feature_index"]]
                X_split = X_split @ node["random_filter"]
            else:
                X_split = X[:, node["random_feature_index"]]
            # leaf samples
            min_samples_left, min_samples_right = random.sample(sample_leaf_list, 2)
            # split_options
            split_options = (X_split, min_samples_left, min_samples_right)
            # Perform threshold split search only if node has not hit max depth
            if self.max_leafs is None:
                max_leafs_condition = True
            else:
                max_leafs_condition = (self.leaf_num < self.max_leafs)
            if max_leafs_condition and (N >= (min_samples_left + min_samples_right)):
                # feature_threshold
                feature_threshold = []
                for j_feature in range(X_split.shape[1]):
                    threshold_search = []
                    for i in range(N):
                        threshold_search.append(X_split[i, j_feature])  # round
                    threshold_search = list(set(threshold_search))

                    if len(threshold_search) > NUM_SPLIT:
                        value_min, value_max = min(threshold_search), max(threshold_search)
                        threshold_search = np.linspace(value_min, value_max, num=NUM_SPLIT)
                        threshold_search = list(np.around(threshold_search, decimals=3))
                        threshold_search = list(set(threshold_search))

                    for threshold in threshold_search:
                        idx_left, idx_right = split_idx(j_feature, threshold, X_split)
                        if (len(idx_left) >= min_samples_left) and (len(idx_right) >=
                                                                    min_samples_right):
                            feature_threshold.append((j_feature, threshold))
                # random threshold
                N_threshold = len(feature_threshold)
                idx = random.sample(range(N_threshold), ceil(sqrt(N_threshold)))
                feature_threshold_random = []
                for i in idx:
                    feature_threshold_random.append(feature_threshold[i])
                feature_threshold = feature_threshold_random
                # Parallel Split
                self.parallel_data = data
                split_result_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._split_loss)(f_t, split_options) for f_t in feature_threshold)
                self.parallel_data = None
                # Get split feature and threshold
                loss_list = [x[0] for x in split_result_list]
                loss_temp = [x for x in loss_list if x is not None]
                if loss_temp:
                    best_loss_split = min(loss_temp)
                    best_loss_index = loss_list.index(best_loss_split)
                    j_feature, threshold = feature_threshold[best_loss_index]
                    _, alpha_left, alpha_right = split_result_list[best_loss_index]
                else:
                    best_loss_split = None
                #
                # Update best parameters if loss is lower
                if (best_loss_split is not None) and (best_loss_split < loss_best):
                    # print(min_samples_left, min_samples_right)
                    # print('alpha_left: ', alpha_left, 'alpha_right: ', alpha_right)
                    idx_left, idx_right = split_idx(j_feature, threshold, X_split)
                    data_left, data_right = split_data(data, idx_left, idx_right)
                    # Compute weight loss function
                    loss_left, model_left, data_left = self._fit_model(data_left, alpha_left)
                    loss_right, model_right, data_right = self._fit_model(data_right, alpha_right)
                    did_split = True
                    models_best = [model_left, model_right]
                    data_best = [data_left, data_right]
                    loss_best = [loss_left, loss_right]
                    N_left_right = [len(idx_left), len(idx_right)]

        # Return the best result
        result = {
            "did_split": did_split,
            "models": models_best,
            "data": data_best,
            "j_feature": j_feature,
            "threshold": threshold,
            "loss": loss_best,
            "N_left_right": N_left_right
        }

        return result

    def _split_loss(self, f_t, split_options):
        # Split data based on threshold
        j_feature, threshold = f_t
        X_split, min_samples_left, min_samples_right = split_options
        idx_left, idx_right = split_idx(j_feature, threshold, X_split)
        data_left, data_right = split_data(self.parallel_data, idx_left, idx_right)
        N_left, N_right = len(idx_left), len(idx_right)
        N = N_left + N_right
        # Splitting conditions
        split_conditions = [N_left >= min_samples_left, N_right >= min_samples_right]
        # Do not attempt to split if split conditions not satisfied
        if not all(split_conditions):
            return None, None, None

        # Compute weight loss function
        loss_left, model_left, _ = self._fit_model(data_left)
        loss_right, model_right, _ = self._fit_model(data_right)
        loss_split = (N_left * loss_left + N_right * loss_right) / N
        # L2
        if (self.task == 'reg') or (self.n_classes == 2):
            coef_left = model_left.model.coef_
            alpha_left = model_left.alpha
            coef_right = model_right.model.coef_
            alpha_right = model_right.alpha
        else:
            coef_left = np.array([m.model.coef_ for m in model_left])
            alpha_left = model_left[0].alpha
            coef_right = np.array([m.model.coef_ for m in model_right])
            alpha_right = model_right[0].alpha

        L2_left = np.linalg.norm(coef_left)**2
        L2_right = np.linalg.norm(coef_right)**2
        loss_alpha = (alpha_left * L2_left + alpha_right * L2_right) / N
        loss_split += loss_alpha
        return loss_split, alpha_left, alpha_right

    def _fit_model(self, data, alpha=None):
        (X, y, output) = data
        X_continus = np.delete(X, self.categorical_feature_index, axis=1)
        if alpha is None:
            alpha = random.sample(self.L2_list, 1)[0]
        model = weight_ridge(alpha)
        if self.task == 'clf':
            prob = self.output2prob(output)
            if self.n_classes == 2:

                weight, z = self._weight_and_response(y, prob)
                X_train, z_train, weight_train = filter_quantile(
                    X_continus, z, weight, trim_quantile=0.05)
                new_estimators = deepcopy(model)  # must deepcopy the model!
                new_estimators.fit(X_train, z_train, weight_train)
            else:

                new_estimators = []
                for j in range(self.n_classes):
                    # weight
                    weight, z = self._weight_and_response(y[:, j], prob[:, j])
                    # filter
                    X_train, z_train, weight_train = filter_quantile(
                        X_continus, z, weight, trim_quantile=0.05)
                    model_copy = deepcopy(model)  # must deepcopy the model!
                    model_copy.fit(X_train, z_train, weight_train)
                    new_estimators.append(model_copy)
            new_scores = self.predict_score(new_estimators, X)
            y_pred = new_scores + output
            prob = self.output2prob(y_pred)
            loss = self.loss(y, prob)
        else:
            targets = y - output
            new_estimators = deepcopy(model)
            X_train, targets = filter_quantile_high(X_continus, targets, 0.05)
            new_estimators.fit(X_train, targets)
            new_scores = self.predict_score(new_estimators, X)
            y_pred = new_scores + output
            loss = self.loss(y, y_pred)

        assert loss >= 0.0
        return loss, new_estimators, (X, y, y_pred)

    def _weight_and_response(self, y, prob):
        sample_weight = prob * (1. - prob)
        sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)
        with np.errstate(divide="ignore", over="ignore"):
            z = np.where(y, 1. / prob, -1. / (1. - prob))
        z = np.clip(z, a_min=-max_response, a_max=max_response)
        return sample_weight, z

    def _create_node(self, data, depth, container, loss_node, model_node=None):
        #
        X, y, output = data
        split_feature_demension = self.split_feature_demension
        #
        feature_candidate = []
        for j_feature in range(self.feature_demension):
            if len(np.unique(X[:, j_feature])) >= 2:
                feature_candidate.append(j_feature)
        if self.RC == 'T':
            if split_feature_demension <= len(feature_candidate):
                random_feature_index = random.sample(feature_candidate, split_feature_demension)
                random_filter = np.random.rand(split_feature_demension, split_feature_demension)
            elif len(feature_candidate) > 0:
                random_feature_index = feature_candidate
                random_filter = np.random.rand(len(feature_candidate), len(feature_candidate))
            else:
                random_feature_index = None
                random_filter = None
        else:
            random_filter = None
            if split_feature_demension <= len(feature_candidate):
                random_feature_index = random.sample(feature_candidate, split_feature_demension)
            elif len(feature_candidate) > 0:
                random_feature_index = feature_candidate
            else:
                random_feature_index = None
        #
        node = {
            "name": "node",
            "index": container["index_node_global"],
            "loss": loss_node,
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
        container["index_node_global"] += 1
        return node


# ***********************************
#
# Side functions
#
# ***********************************


def filter_quantile(X, z, sample_weight, trim_quantile):
    threshold = np.quantile(sample_weight, trim_quantile, interpolation="lower")
    mask = (sample_weight >= threshold)
    X_train = X[mask]
    z_train = z[mask]
    sample_weight = sample_weight[mask]
    return X_train, z_train, sample_weight


def filter_quantile_high(X, y, trim_quantile):
    y_abs = np.abs(y)
    threshold_high = np.quantile(y_abs, 1 - trim_quantile, interpolation="lower")
    mask = (y_abs <= threshold_high)
    X_train = X[mask]
    y_train = y[mask]
    return X_train, y_train


def split_idx(j_feature, threshold, X_split):
    idx_left = np.where(X_split[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X_split)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X_split)
    return idx_left, idx_right


def split_data(data, idx_left, idx_right):
    X, y, output = data
    data_left = (X[idx_left], y[idx_left], output[idx_left])
    data_right = (X[idx_right], y[idx_right], output[idx_right])
    return data_left, data_right
