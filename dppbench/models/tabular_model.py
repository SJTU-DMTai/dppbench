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
        pred = np.asarray(self.predict(X)).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if len(y) == 0:
            raise ValueError("cannot evaluate an empty target")
        if len(pred) != len(y):
            raise ValueError(
                f"prediction length {len(pred)} does not match target length {len(y)}"
            )
        try:
            bad_y = ~np.isfinite(y.astype(float))
            bad_pred = ~np.isfinite(pred.astype(float))
        except (TypeError, ValueError) as exc:
            raise ValueError("target and predictions must be numeric") from exc
        if bad_y.any():
            raise ValueError(f"target contains {int(bad_y.sum())} non-finite values")
        if bad_pred.any():
            raise ValueError(
                f"predictions contain {int(bad_pred.sum())} non-finite values"
            )
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
                if m == "auc" and len(np.unique(y)) < 2:
                    raise ValueError(
                        "auc requires at least two target classes; "
                        f"observed {np.unique(y).tolist()}"
                    )
                result[m] = float(fn(y, pred))
        return result

    def feature_importance(self):
        raise NotImplementedError
