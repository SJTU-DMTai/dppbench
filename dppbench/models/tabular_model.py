import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score,
)


class TabularModel:
    def __init__(self, task="binary", seed=42):
        self.task = task
        self.seed = seed
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y, metrics=None):
        pred = self.predict(X)
        metrics = metrics or (["auc"] if self.task == "binary" else ["mse"])
        metric_map = {
            "auc": roc_auc_score,
            "logloss": log_loss,
            "mse": mean_squared_error,
            "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
            "mae": mean_absolute_error,
            "r2": r2_score,
        }
        result = {}
        for m in metrics:
            fn = metric_map.get(m)
            if fn:
                try:
                    result[m] = float(fn(y, pred))
                except (ValueError, ZeroDivisionError):
                    result[m] = float("nan")
        return result

    def feature_importance(self):
        raise NotImplementedError
