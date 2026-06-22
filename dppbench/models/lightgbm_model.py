import numpy as np
import lightgbm as lgb
from .tabular_model import TabularModel


class LightGBMModel(TabularModel):
    def __init__(self, task="binary", seed=42, **params):
        super().__init__(task=task, seed=seed)
        self.params = params
        self.model = None
        self.feature_names = None

    def _get_lgb_params(self):
        default_params = {
            "objective": "binary" if self.task == "binary" else "regression",
            "metric": "auc" if self.task == "binary" else "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 34,
            "max_depth": 5,
            "min_child_samples": 50,
            "colsample_bytree": 0.3,
            "subsample": 0.8,
            "subsample_freq": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": self.seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        default_params.update(self.params)
        return default_params

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            categorical_features=None, sample_weight=None,
            eval_sample_weight=None, **kwargs):
        params = self._get_lgb_params()
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=100),
        ]

        if self.task == "binary":
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None
        cat_feats = categorical_features or "auto"

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight,
            categorical_feature=cat_feats,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        if self.task == "binary":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
