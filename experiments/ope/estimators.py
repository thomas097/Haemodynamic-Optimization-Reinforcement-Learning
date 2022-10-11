import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor


class LassoRegression:
    def __init__(self, alpha=0.01):
        self._model = Lasso(alpha=alpha, fit_intercept=True, random_state=1)

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)  # catch convergence warnings
            self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class RandomForest(LassoRegression):
    def __init__(self):
        super().__init__()
        self._model = RandomForestRegressor(n_estimators=150, max_depth=7, random_state=1)
