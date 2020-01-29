"""

 ModelTree.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
from copy import deepcopy
from graphviz import Digraph
from sklearn.linear_model import LinearRegression, Ridge


class PatchTree(object):
    def __init__(self,
                 model,
                 max_depth=5,
                 min_samples_leaf=10,
                 search_type="greedy",
                 n_search_grid=100):

        self.model = model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.search_type = search_type
        self.n_search_grid = n_search_grid
        self.tree = None

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

        if verbose:
            print(
                " max_depth={}, min_samples_leaf={}, search_type={}...".format(
                    max_depth, min_samples_leaf, search_type))

        def _build_tree(X, y):

            global index_node_global

            def _create_node(X, y, depth, container, weight, output):
                loss_node, model_node, output = _fit_model(
                    X, y, model, weight, output)
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

            def _create_node2(X, y, depth, container, weight, output,
                              loss_node, model_node):
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

            # Recursively split node + traverse node until a terminal node is reached
            def _split_traverse_node(node, container, node_temp, loss_temp,
                                     patch_num):

                # Perform split and collect result
                result = _splitter(
                    node,
                    model,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    search_type=search_type,
                    n_search_grid=n_search_grid,
                    patch_num=patch_num)

                # Return terminal node if split is not advised
                if not result["did_split"]:
                    # print('patch_num: ', patch_num)
                    if max(loss_temp) == 0.000001:
                        return
                    if verbose:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(
                            " {}*leaf {} @ depth {}: loss={:.6f}, N={}".format(
                                depth_spacing_str, node["index"],
                                node["depth"], node["loss"], result["N"]))
                    if patch_num < max_depth:
                        loss_temp[node["index"]] = 0.000001
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
                node["children"]["left"] = _create_node2(
                    X_left, y_left, node["depth"] + 1, container, weight_left,
                    output_left, loss_left, model_left)
                node_temp.append(node["children"]["left"])
                loss_temp.append(node["children"]["left"]["n_samples"] *
                                 node["children"]["left"]["loss"])
                node["children"]["right"] = _create_node2(
                    X_right, y_right, node["depth"] + 1, container,
                    weight_right, output_right, loss_right, model_right)
                node_temp.append(node["children"]["right"])
                loss_temp.append(node["children"]["right"]["n_samples"] *
                                 node["children"]["right"]["loss"])
                patch_num = len(loss_temp) - loss_temp.count(0)
                #decide split node
                split_index = loss_temp.index(max(loss_temp))
                _split_traverse_node(node_temp[split_index], container,
                                     node_temp, loss_temp, patch_num)

            weight = (1 / len(X)) * np.ones(len(X))
            output = np.zeros(len(X))
            container = {"index_node_global": 0}  # mutatable container
            root = _create_node(X, y, 0, container, weight,
                                output)  # depth 0 root node
            node_temp = []
            node_temp.append(root)
            loss_temp = []
            loss_temp.append(root["n_samples"] * root["loss"])
            patch_num = len(loss_temp) - loss_temp.count(0)
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

        def _predict(node, x, y_pred_x=0):
            no_children = node["children"]["left"] is None and node["children"]["right"] is None
            if no_children:
                y_pred_x += node["model"].predict([x])[0]
                return y_pred_x
            else:
                y_pred_x += node["model"].predict([x])[0]
                if x[node["j_feature"]] <= node["threshold"]:  # x[j] < threshold
                    return _predict(node["children"]["left"], x, y_pred_x)
                else:  # x[j] > threshold
                    return _predict(node["children"]["right"], x, y_pred_x)

        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred

    # ======================
    # Loss
    # ======================
    def loss(self, X, y, y_pred):
        loss = self.model.loss(y, y_pred)
        return loss

    # ======================
    # Tree diagram
    # ======================
    def export_graphviz(self,
                        output_filename,
                        feature_names,
                        export_png=True,
                        export_pdf=False):
        """
         Assumes node structure of:

           node["name"]
           node["n_samples"]
           node["children"]["left"]
           node["children"]["right"]
           node["j_feature"]
           node["threshold"]
           node["loss"]

        """
        g = Digraph('g', node_attr={'shape': 'record', 'height': '.1'})

        def build_graphviz_recurse(node,
                                   parent_node_index=0,
                                   parent_depth=0,
                                   edge_label=""):

            # Empty node
            if node is None:
                return

            # Create node
            node_index = node["index"]
            if node["children"]["left"] is None and node["children"]["right"] is None:
                threshold_str = ""
            else:
                threshold_str = "{} <= {:.1f}\\n".format(
                    feature_names[node['j_feature']], node["threshold"])

            label_str = "{} n_samples = {}\\n loss = {:.6f}\\n total_loss = {:.6f}".format(
                threshold_str, node["n_samples"], node["loss"],
                node["n_samples"] * node["loss"])

            # Create node
            nodeshape = "rectangle"
            bordercolor = "black"
            fillcolor = "white"
            fontcolor = "black"
            g.attr('node', label=label_str, shape=nodeshape)
            g.node(
                'node{}'.format(node_index),
                color=bordercolor,
                style="filled",
                fillcolor=fillcolor,
                fontcolor=fontcolor)

            # Create edge
            if parent_depth > 0:
                g.edge(
                    'node{}'.format(parent_node_index),
                    'node{}'.format(node_index),
                    label=edge_label)

            # Traverse child or append leaf value
            build_graphviz_recurse(
                node["children"]["left"],
                parent_node_index=node_index,
                parent_depth=parent_depth + 1,
                edge_label="")
            build_graphviz_recurse(
                node["children"]["right"],
                parent_node_index=node_index,
                parent_depth=parent_depth + 1,
                edge_label="")

        # Build graph
        build_graphviz_recurse(
            self.tree, parent_node_index=0, parent_depth=0, edge_label="")

        # Export pdf
        if export_pdf:
            print("Saving model tree diagram to '{}.pdf'...".format(
                output_filename))
            g.format = "pdf"
            g.render(filename=output_filename, view=False, cleanup=True)

        # Export png
        if export_png:
            print("Saving model tree diagram to '{}.png'...".format(
                output_filename))
            g.format = "png"
            g.render(filename=output_filename, view=False, cleanup=True)


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
              patch_num=5):

    # Extract data
    X, y = node["data"]
    depth = node["depth"]
    N, d = X.shape
    weight = node["weight"]
    output = node["output"]
    weight = weight * np.exp(-y * output)
    weight = weight / np.sum(weight)
    # Find feature splits that might improve loss
    did_split = False
    loss_best = node["loss"]
    data_best = None
    models_best = None
    j_feature_best = None
    threshold_best = None

    # Perform threshold split search only if node has not hit max depth
    if (patch_num >= 0) and (patch_num < max_depth):
        for j_feature in range(d):
            threshold = decide_threshold(X[:, j_feature], y, weight)[0]
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

    return result


def _fit_model(X, y, model, weight, output):
    model_copy = deepcopy(model)  # must deepcopy the model!
    model_copy.fit(X, y, weight)
    y_pred = model_copy.predict(X)
    y_pred = y_pred + output
    loss = model_copy.loss(y, y_pred)
    assert loss >= 0.0
    return loss, model_copy, y_pred


def _split_data(j_feature, threshold, X, y, weight, output):
    idx_left = np.where(X[:, j_feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return (X[idx_left], y[idx_left], weight[idx_left],
            output[idx_left]), (X[idx_right], y[idx_right], weight[idx_right],
                                output[idx_right])


def decide_threshold(X, y, weight):
    # m = Ridge(alpha=0.1)
    m = LinearRegression()
    X = np.reshape(X, (len(X), -1))
    m.fit(X, y, weight)
    return -m.intercept_ / (m.coef_ + 0.0001)
