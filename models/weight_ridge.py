"""

 weight_ridge.py 

"""
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


class weight_ridge:
    def __init__(self, alpha):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)
        self.alpha = alpha

    def fit(self, X, y, weight=None):
        self.model.fit(X, y, weight)

    def predict(self, X):
        if X.ndim == 2:
            return self.model.predict(X)
        else:
            return self.model.predict(X.reshape(1, -1))
