"""

 Regression Boosting Tree

"""

import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, mean_squared_error
from math import ceil, sqrt
import random


class PatchTree(object):
    def __init__(self,
                 model,
                 max_depth=5,
                 min_samples_leaf=10,
                 search_type="greedy",
                 n_search_grid=100,
                 categorical_feature_index=None):

        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.search_type = search_type
        self.n_search_grid = n_search_grid
        self.tree = None
        self.TotalToss = []
        self.X = None
        self.y = None
        self.categorical_feature_index = categorical_feature_index

    def get_params(self, deep=True):
        return {
            "model": self.model.get_params() if deep else self.model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "search_type": self.search_type,
            "n_search_grid": self.n_search_grid,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(
            class_name,
            ', '.join([
                "{}={}".format(k, v)
                for k, v in self.get_params(deep=False).items()
            ]))

    # ======================
    # Fit
    # ======================
    def fit(self, X, y, verbose=False):

        # Settings
        model = self.model
        min_samples_leaf = self.min_samples_leaf
        max_depth = self.max_depth
        search_type = self.search_type
        n_search_grid = self.n_search_grid
        self.X = X
        self.y = y
        if verbose:
            print(
                " max_depth={}, min_samples_leaf={}, search_type={}...".format(
                    max_depth, min_samples_leaf, search_type))

        def _build_tree(X, y):

            # Recursively split node + traverse node until a terminal node is reached
            def _split_traverse_node(node, container, node_temp, loss_temp,
                                     patch_num):
                """
                    main loop
                """
                # Perform split and collect result
                result, loss_decrease = _splitter(
                    node,
                    model,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    search_type=search_type,
                    n_search_grid=n_search_grid,
                    patch_num=patch_num,
                    categorical_feature_index=self.categorical_feature_index)

                # Return terminal node if split is not advised
                if not result["did_split"]:
                    # print('patch_num: ', patch_num)
                    if max(loss_temp) == 0:
                        return
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(
                            " {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(
                                depth_spacing_str, node["index"],
                                node["depth"], node["loss"], result["N"]))
                    if patch_num < max_depth:
                        loss_temp[node["index"]] = 0
                        split_index = loss_temp.index(max(loss_temp))
                        _split_traverse_node(node_temp[split_index], container,
                                             node_temp, loss_temp, patch_num)
                    return

                # Update node information based on splitting result
                node["j_feature"] = result["j_feature"]
                node["threshold"] = result["threshold"]
                del node["data"]  # delete node stored data

                # Extract splitting results
                (X_left, y_left, weight_left, output_left,
                 loss_left), (X_right, y_right, weight_right, output_right,
                              loss_right) = result["data"]
                weight_left = weight_left / np.sum(weight_left)
                weight_right = weight_right / np.sum(weight_right)
                model_left, model_right = result["models"]

                # Report created node to user
                if verbose:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(
                        " {}node {} @ depth {}: loss={:.6f}, j_feature={}, threshold={:.6f}, N=({},{})".
                        format(depth_spacing_str, node["index"], node["depth"],
                               node["loss"], node["j_feature"],
                               node["threshold"], len(X_left), len(X_right)))
                loss_temp[node["index"]] = 0
                # Create children nodes
                node["children"]["left"] = _create_node(
                    X_left, y_left, node["depth"] + 1, container, weight_left,
                    output_left, loss_left, model_left)
                node_temp.append(node["children"]["left"])
                loss_temp.append(node["children"]["left"]["n_samples"] *
                                 node["children"]["left"]["loss"])
                node["children"]["right"] = _create_node(
                    X_right, y_right, node["depth"] + 1, container,
                    weight_right, output_right, loss_right, model_right)
                node_temp.append(node["children"]["right"])
                loss_temp.append(node["children"]["right"]["n_samples"] *
                                 node["children"]["right"]["loss"])
                patch_num += 1

                y_pred = np.array([
                    _predict(root, x, 0, self.categorical_feature_index)
                    for x in self.X
                ])
                self.TotalToss.append(self.loss(self.y, y_pred))
                # decide split node
                split_index = loss_temp.index(max(loss_temp))
                _split_traverse_node(node_temp[split_index], container,
                                     node_temp, loss_temp, patch_num)

            weight = (1 / len(X)) * np.ones(len(X))
            output = np.zeros(len(X))
            container = {"index_node_global": 0}  # mutatable container
            loss_node, model_node, output = _fit_model(X, y, model, weight,
                                                       output)
            root = _create_node(X, y, 0, container, weight, output, loss_node,
                                model_node)  # depth 0 root node
            self.TotalToss.append(loss_node)
            node_temp = []
            node_temp.append(root)
            loss_temp = []
            loss_temp.append(root["n_samples"] * root["loss"])
            patch_num = 1
            _split_traverse_node(root, container, node_temp, loss_temp,
                                 patch_num)  # split and traverse root node

            return root

        # Construct tree
        self.tree = _build_tree(X, y)

    # ======================
    # Predict
    # ======================
    def predict(self, X):
        assert self.tree is not None
        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred

    # ======================
    # Loss
    # ======================
    def loss(self, y, y_pred):
        loss = mean_squared_error(y, y_pred)
        return loss


# ***********************************
#
# Side functions
#
# ***********************************


def _splitter(node,
              model,
              max_depth=5,
              min_samples_leaf=10,
              search_type="greedy",
              n_search_grid=100,
              patch_num=5,
              categorical_feature_index=None):
    """
    Split the node and collect result
    """
    # Extract data
    X, y = node["data"]
    # depth = node["depth"]
    N, d = X.shape
    weight = node["weight"]
    output = node["output"]
    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    data_best = None
    models_best = None
    j_feature_best = None
    threshold_best = None

    # Perform threshold split search only if node has not hit max depth
    if (patch_num >= 0) and (patch_num < max_depth):
        random_j_feature = random.sample(range(d), ceil(sqrt(d)))
        for j_feature in random_j_feature:

            # If using adaptive search type, decide on one to use
            search_type_use = search_type
            if search_type == "adaptive":
                if N > n_search_grid:
                    search_type_use = "grid"
                else:
                    search_type_use = "greedy"

            # Use decided search type and generate threshold search list (j_feature)
            threshold_search = []
            if search_type_use == "greedy":
                for i in range(N):
                    threshold_search.append(X[i, j_feature])
            elif search_type_use == "grid":
                x_min, x_max = np.min(X[:, j_feature]), np.max(X[:, j_feature])
                dx = (x_max - x_min) / n_search_grid
                for i in range(n_search_grid + 1):
                    threshold_search.append(x_min + i * dx)
            else:
                exit(
                    "err: invalid search_type = {} given!".format(search_type))

            # Perform threshold split search on j_feature
            for threshold in threshold_search:

                # Split data based on threshold
                (X_left, y_left, weight_left,
                 output_left), (X_right, y_right, weight_right,
                                output_right) = _split_data(
                                    j_feature, threshold, X, y, weight, output)
                N_left, N_right = len(X_left), len(X_right)

                # Splitting conditions
                split_conditions = [
                    N_left >= min_samples_leaf, N_right >= min_samples_leaf
                ]

                # Do not attempt to split if split conditions not satisfied
                if not all(split_conditions):
                    continue

                # Compute weight loss function
                loss_left, model_left, output_left = _fit_model(
                    X_left, y_left, model, weight_left, output_left)
                loss_right, model_right, output_right = _fit_model(
                    X_right, y_right, model, weight_right, output_right)
                loss_split = (N_left * loss_left + N_right * loss_right) / N

                # Update best parameters if loss is lower
                if loss_split < loss_best:
                    did_split = True
                    loss_best = loss_split
                    models_best = [model_left, model_right]
                    data_best = [(X_left, y_left, weight_left, output_left,
                                  loss_left), (X_right, y_right, weight_right,
                                               output_right, loss_right)]
                    j_feature_best = j_feature
                    threshold_best = threshold

    # Return the best result
    result = {
        "did_split": did_split,
        "loss": loss_best,
        "models": models_best,
        "data": data_best,
        "j_feature": j_feature_best,
        "threshold": threshold_best,
        "N": N
    }
    loss_decrease = (loss_best - node["loss"]) * N
    return result, loss_decrease


def _fit_model(X, y, model, weight, output):
    d = X.shape[1]
    random_feature_index = random.sample(range(d), ceil(sqrt(d)))
    X_random = X[:, random_feature_index]
    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.feature_index = random_feature_index
    model_copy.fit(X_random, y - output, weight)
    y_pred = model_copy.predict(X)
    y_pred = y_pred + output
    loss = mean_squared_error(y, y_pred)
    assert loss >= 0.0
    return loss, model_copy, y_pred


def _split_data(j_feature, threshold, X, y, weight, output):
    idx_left = np.where(X[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return (X[idx_left], y[idx_left], weight[idx_left],
            output[idx_left]), (X[idx_right], y[idx_right], weight[idx_right],
                                output[idx_right])


def _create_node(X, y, depth, container, weight, output, loss_node,
                 model_node):
    node = {
        "name": "node",
        "index": container["index_node_global"],
        "loss": loss_node,
        "model": model_node,
        "data": (X, y),
        "n_samples": len(X),
        "j_feature": None,
        "threshold": None,
        "children": {
            "left": None,
            "right": None
        },
        "depth": depth,
        "weight": weight,
        "output": output
    }
    container["index_node_global"] += 1
    return node


def _predict(node, x, y_pred_x=0, categorical_feature_index=None):
    if categorical_feature_index is not None:
        x_continus = x[categorical_feature_index]
    else:
        x_continus = x
    no_children = node["children"]["left"] is None and node["children"]["right"] is None
    if no_children:
        y_pred_x += node["model"].predict(x_continus)[0]
        return y_pred_x
    else:
        y_pred_x += node["model"].predict(x_continus)[0]
        if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
            return _predict(node["children"]["left"], x, y_pred_x,
                            categorical_feature_index)
        else:  # x[j] > threshold
            return _predict(node["children"]["right"], x, y_pred_x,
                            categorical_feature_index)
