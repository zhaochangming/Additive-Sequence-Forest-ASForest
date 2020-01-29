import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from functools import partial


class TreeNode(object):
    def __init__(self, element, childs=dict(), info=None):
        self.element = element
        self.childs = childs
        self.info = info


class SimpleSoftTree(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 penalty='l2',
                 C=1.0,
                 solver='liblinear',
                 multi_class='ovr',
                 base_max_iter=100,
                 max_depth=1,
                 min_samples_split=50,
                 max_purity_split=1.,
                 n_jobs=-1):
        """
        Use Logistic Regression as the base model for every node in a Tree.
        :param penalty: see sklearn.linear_model.LogisticRegression
        :param C: see sklearn.linear_model.LogisticRegression
        :param solver: see sklearn.linear_model.LogisticRegression
        :param multi_class: see sklearn.linear_model.LogisticRegression
        :param base_max_iter: see sklearn.linear_model.LogisticRegression
        :param max_depth: The maximum depth of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param max_purity_split: Then maximum purity required to split an internal node.
        :param n_jobs: number of processors. -1 means use all the processors.
        """
        self.penalty = penalty
        self.C = C
        self.base_max_iter = base_max_iter
        self.solver = solver
        self.multi_class = multi_class
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_purity_split = max_purity_split
        self.n_jobs = n_jobs
        self.classes = None
        self.nb_class = None
        self._count = {}
        self.clf_number = 0
        self.info = []

    def fit(self, X, y, initial_class_weight=None):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.nb_class = len(self.classes)
        self.start_node = self._create_Tree(
            X, y, initial_class_weight, depth_id=0)
        return self

    def _base_classfier(self, X, y, class_weight):
        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            class_weight=class_weight,
            solver=self.solver,
            max_iter=self.base_max_iter,
            multi_class=self.multi_class).fit(X, y)

    def _create_Tree(self,
                     X,
                     y,
                     class_weight,
                     father_node_name=None,
                     father_class=None,
                     depth_id=0):
        if depth_id not in self._count.keys():
            self.info.append([])
            self._count[depth_id] = 1
        else:
            self._count[depth_id] += 1
        # print('fitting:depth={}, node_id={}'.format(depth_id, self._count[depth_id] - 1))
        clf = self._base_classfier(X, y, class_weight)
        label = clf.predict(X)
        self.clf_number += 1
        label_unique = np.unique(label)
        node_info = {
            'name': "{}_{}".format(depth_id, self._count[depth_id] - 1),
            'depth_id': depth_id,
            'node_id': self._count[depth_id] - 1,
            'father_class': father_class,
            'father_node_name': father_node_name,
            'sample_per_label': {
                c: int(np.sum(np.where(label == c, 1, 0.)))
                for c in label_unique
            }
        }
        self.info[depth_id].append(node_info)

        if depth_id + 1 >= self.max_depth:
            childs = {c: None for c in self.classes}
        else:
            childs = dict()
            # check depth

            for i in self.classes:
                _X, _y = X[np.argwhere(label == i).ravel(), :], y[np.argwhere(
                    label == i).ravel()]

                # check min_sample_split
                if (len(_y) <= self.min_samples_split) or (len(_y) == 0):
                    childs[i] = None
                    continue

                _class_weight = {
                    c: len(_y) / np.sum(np.where(_y == c, 1, 0)).astype(
                        np.float64)
                    for c in np.unique(_y)
                }

                # check purity
                if min(_class_weight.values()) <= 1. / self.max_purity_split:
                    childs[i] = None
                    continue

                childs[i] = self._create_Tree(
                    X=_X,
                    y=_y,
                    class_weight=_class_weight,
                    father_node_name=node_info['name'],
                    father_class=i,
                    depth_id=depth_id + 1)

        return TreeNode(element=clf, childs=childs, info=node_info)

    @staticmethod
    def _predict_one_example(x, start_node):
        x = np.reshape(x, newshape=(1, -1))
        node = start_node
        y = node.element.predict(x)[0]
        while node.childs[y] is not None:
            node = node.childs[y]
            y = node.element.predict(x)[0]
        return y

    def predict(self, X):
        if self.n_jobs == -1:
            pool = Pool(processes=None)
        else:
            pool = Pool(processes=self.n_jobs)
        y = np.array(
            pool.map(
                partial(self._predict_one_example, start_node=self.start_node),
                X))
        return y

    def show_tree(self):
        print('format: depth Depth: NODE_ID(FATHER_ID-CLASS)-{LABEL:NUMBERS}')
        for depth_id in range(len(self.info)):
            print('depth {}:'.format(depth_id), end=' ')
            depth_info = self.info[depth_id]
            for node_info in depth_info:
                print(
                    "{}({}-{})-{}".format(node_info['node_id'],
                                          node_info['father_node_name'],
                                          node_info['father_class'],
                                          node_info['sample_per_label']),
                    end='    ')
            print()
