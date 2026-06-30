import os
import sys
import yaml
import time
import tempfile
import traceback
import shutil
import copy
import pandas as pd

# Repo root (= parent of baselines/) so ``import dppbench`` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


DATASET_REGISTRY = {
    # --- tabular binary classification ---------------------------------
    "berka": ("tabular", "dppbench.tasks.berka.berka_data.BerkaData"),
    "bondora": ("tabular", "dppbench.tasks.bondora.bondora_data.BondoraData"),
    "default_credit": (
        "tabular",
        "dppbench.tasks.default_credit.default_credit_data.DefaultCreditData",
    ),
    "fraud_detection": (
        "tabular",
        "dppbench.tasks.fraud_detection.fraud_detection_data.FraudDetectionData",
    ),
    "home_credit": (
        "tabular",
        "dppbench.tasks.home_credit.home_credit_data.HomeCreditData",
    ),
    "polish_bankruptcy": (
        "tabular",
        "dppbench.tasks.polish_bankruptcy.polish_bankruptcy_data.PolishBankruptcyData",
    ),
    # --- tabular time-series regression --------------------------------
    "beijing_air_quality": (
        "tabular",
        "dppbench.tasks.beijing_air_quality.beijing_air_quality_data.BeijingAirQualityData",
    ),
    "bike_sharing": (
        "tabular",
        "dppbench.tasks.bike_sharing.bike_sharing_data.BikeSharingData",
    ),
    "citibike_jc_hourly": (
        "tabular",
        "dppbench.tasks.citibike_jc_hourly.citibike_jc_hourly_data.CitibikeJcHourlyData",
    ),
    "nyc_taxi_hourly": (
        "tabular",
        "dppbench.tasks.nyc_taxi_hourly.nyc_taxi_hourly_data.NycTaxiHourlyData",
    ),
    # --- graph node classification ------------------------------------
    "elliptic_bitcoin": (
        "graph",
        "dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data.EllipticBitcoinData",
    ),
    # --- recommendation -----------------------------------------------
    "amazon_beauty": (
        "rec",
        "dppbench.tasks.amazon_beauty.amazon_beauty_data.AmazonBeautyData",
    ),
    "kuairec": (
        "rec",
        "dppbench.tasks.kuairec.kuairec_data.KuairecData",
    ),
    "movielens": (
        "rec",
        "dppbench.tasks.movielens.movielens_data.MovielensData",
    ),
    "yelp": ("rec", "dppbench.tasks.yelp.yelp_data.YelpData"),
    "tenrec": ("rec", "dppbench.tasks.tenrec.tenrec_data.TenrecData"),
}


def _import_class(dotted_path):
    import importlib
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Default search-time fast-training overrides. Each key targets a class of
# downstream model; values are merged into the appropriate cfg section right
# before model construction, leaving the on-disk task ``model.yaml`` untouched.
DEFAULT_FAST_TRAIN_OVERRIDES = {
    "lightgbm": {"n_estimators": 100},
    "rec": {"epochs": 2, "early_stopping_patience": 1},
    "graph": {"epochs": 10},
    "seq": {"epochs": 2, "early_stopping_patience": 1},
    "tabular_deep": {"epochs": 2, "max_train_rows": 5000, "max_features": 64, "batch_size": 256},
}

REC_MODELS = {"FNN", "DeepFM", "DIN", "DIEN", "SIM"}
TABULAR_DEEP_MODELS = {"MLP", "TabTransformer", "FTTransformer", "SAINT"}
SEQUENCE_MODELS = {"LSTM", "GRU", "Transformer"}
GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT"}


class TrainingExecutor:
    def __init__(self, task_dir, data_name, data_dir=None, device="cpu",
                 model_name=None, model_config=None):
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.original_yaml = os.path.join(task_dir, "pre_process.yaml")
        self.model_yaml = os.path.join(task_dir, "model.yaml")
        self._data = None

        if data_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATASET_REGISTRY.keys())}")
        self._task_type, self._data_cls_path = DATASET_REGISTRY[data_name]

        self._work_dir = tempfile.mkdtemp(prefix="preproc_opt_")
        self._working_yaml = os.path.join(self._work_dir, "pre_process.yaml")
        shutil.copy2(self.original_yaml, self._working_yaml)

        # Search-time fast mode toggle. When enabled, training rounds
        # (n_estimators / epochs) are reduced via _apply_fast_overrides at the
        # start of each ``run_training()`` call.
        self._fast_mode = False
        self._fast_overrides = None

    def set_fast_mode(self, enabled: bool, overrides: dict = None):
        """Enable/disable search-time fast training overrides.

        When enabled, the next ``run_training()`` call will shrink training
        rounds for the dispatched downstream model (LightGBM n_estimators,
        DIN/GNN/seq epochs, etc.). ``overrides`` may supply a custom dict in
        the same shape as ``DEFAULT_FAST_TRAIN_OVERRIDES``.
        """
        self._fast_mode = bool(enabled)
        self._fast_overrides = overrides

    def _apply_fast_overrides(self, cfg, model_name):
        """Mutate ``cfg`` in-place with fast-mode overrides for ``model_name``."""
        if not self._fast_mode:
            return cfg
        overrides = self._fast_overrides or DEFAULT_FAST_TRAIN_OVERRIDES
        if model_name == "LightGBM":
            cfg.setdefault("model_params", {}).update(overrides.get("lightgbm", {}))
        elif model_name in SEQUENCE_MODELS:
            cfg.setdefault("model_params", {}).update(overrides.get("seq", {}))
        elif model_name in GRAPH_MODELS:
            cfg.setdefault("model_params", {}).update(overrides.get("graph", {}))
        elif model_name in TABULAR_DEEP_MODELS:
            cfg.setdefault("model_params", {}).update(overrides.get("tabular_deep", {}))
        else:
            # Treat anything else (DIN and other rec models) as a rec model:
            # rec training reads epochs/patience from cfg["train"].
            cfg.setdefault("train", {}).update(overrides.get("rec", {}))
        return cfg

    @property
    def task_type(self):
        return self._task_type

    def _make_data_instance(self):
        cls = _import_class(self._data_cls_path)
        return cls(data_dir=self.data_dir)

    def _make_configured_data_instance(self):
        data = self._make_data_instance()
        if hasattr(data, "set_model_config"):
            data.set_model_config(self.get_model_config())
        return data

    def get_current_yaml(self):
        with open(self._working_yaml, "r", encoding="utf-8") as f:
            return f.read()

    def get_original_yaml(self):
        with open(self.original_yaml, "r", encoding="utf-8") as f:
            return f.read()

    def write_yaml(self, content):
        with open(self._working_yaml, "w", encoding="utf-8") as f:
            f.write(content)

    def save_best_yaml(self, content, output_path=None):
        if output_path is None:
            output_path = os.path.join(self._work_dir, "best_pre_process.yaml")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def cleanup(self):
        if os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def get_model_config(self):
        with open(self.model_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return self._resolve_model_config(cfg)

    def _resolve_model_config(self, cfg):
        cfg = copy.deepcopy(cfg)
        if self.model_config:
            return self._merge_model_option(cfg, self.model_name, self.model_config)
        if not self.model_name:
            return cfg
        options = cfg.get("model_options") or {}
        if self.model_name not in options:
            raise ValueError(
                f"Model '{self.model_name}' is not configured for {self.data_name}. "
                f"Available: {sorted(options) or [cfg.get('model')]}"
            )
        return self._merge_model_option(cfg, self.model_name, options[self.model_name])

    @staticmethod
    def _merge_model_option(base_cfg, model_name, option_cfg):
        merged = copy.deepcopy(base_cfg)
        merged["model"] = model_name
        for key, value in (option_cfg or {}).items():
            if key in {"model_params", "train", "feature"} and isinstance(value, dict):
                merged.setdefault(key, {})
                merged[key].update(copy.deepcopy(value))
            elif key != "model":
                merged[key] = copy.deepcopy(value)
        return merged

    def get_data_summary(self):
        data = self._load_data()
        if self._task_type == "tabular":
            return self._tabular_data_summary(data)
        if self._task_type == "graph":
            return self._graph_data_summary(data)
        return self._rec_data_summary(data)

    def _tabular_data_summary(self, data):
        train_df = data.train_df
        return {
            "task_type": "tabular",
            "train_shape": train_df.shape,
            "columns": list(train_df.columns[:30]),
            "dtypes_summary": {str(k): int(v) for k, v in train_df.dtypes.value_counts().items()},
            "null_ratio": float(train_df.isnull().mean().mean()),
            "target_col": data.target_col,
            "target_distribution": {str(k): int(v) for k, v in train_df[data.target_col].value_counts().items()} if data.target_col in train_df.columns else {},
            "auxiliary_tables": {k: list(v.shape) if v is not None else None for k, v in data.auxiliary_dfs.items()},
        }

    def _rec_data_summary(self, data):
        interaction_df = data.interaction_df
        return {
            "task_type": "rec",
            "interaction_shape": interaction_df.shape,
            "columns": list(interaction_df.columns[:30]),
            "col_types": data.col_types,
            "null_ratio": float(interaction_df.isnull().mean().mean()),
            "has_user_df": data.user_df is not None,
            "has_item_df": data.item_df is not None,
            "user_df_shape": list(data.user_df.shape) if data.user_df is not None else None,
            "item_df_shape": list(data.item_df.shape) if data.item_df is not None else None,
        }

    def _graph_data_summary(self, data):
        train_df = data.train_df
        edges = data.auxiliary_dfs.get("edges") if data.auxiliary_dfs else None
        classes = data.auxiliary_dfs.get("classes") if data.auxiliary_dfs else None
        feat_cols = [c for c in train_df.columns if str(c).startswith("feat_")]
        target_col = getattr(data, "target_col", None)
        id_col = getattr(data, "id_col", None)
        target_distribution = {}
        if target_col in train_df.columns:
            target_distribution = {
                str(k): int(v)
                for k, v in train_df[target_col].value_counts(dropna=False).items()
            }
        return {
            "task_type": "graph",
            "train_shape": train_df.shape,
            "columns": list(train_df.columns[:30]),
            "feature_cols": feat_cols[:30],
            "n_feature_cols": len(feat_cols),
            "dtypes_summary": {str(k): int(v) for k, v in train_df.dtypes.value_counts().items()},
            "null_ratio": float(train_df.isnull().mean().mean()),
            "target_col": target_col,
            "id_col": id_col,
            "target_distribution": target_distribution,
            "edges_shape": list(edges.shape) if edges is not None else None,
            "classes_shape": list(classes.shape) if classes is not None else None,
            "auxiliary_tables": {k: list(v.shape) if v is not None else None for k, v in data.auxiliary_dfs.items()},
        }

    def _load_data(self):
        if self._data is None:
            self._data = self._make_configured_data_instance()
            self._data.load_data()
        return self._data

    def run_training(self):
        if self._task_type == "tabular":
            return self._run_tabular_training()
        elif self._task_type == "graph":
            return self._run_graph_training()
        else:
            return self._run_rec_training()

    def _run_tabular_training(self):
        from dppbench.models import (
            LightGBMModel, MLP, TabTransformer, FTTransformer, SAINT,
        )

        start_time = time.time()
        result = {
            "success": False,
            "metrics": {},
            "error": None,
            "train_shape": None,
            "n_features": None,
            "duration_seconds": None,
        }

        try:
            data = self._make_configured_data_instance()
            data.load_data()
            data.run_pre_process(os.path.abspath(self._working_yaml))

            cfg = self.get_model_config()
            model_name = cfg.get("model", "LightGBM")
            self._apply_fast_overrides(cfg, model_name)
            train_cfg = cfg.get("train", {})
            model_params = cfg.get("model_params", {})

            splits = data.split(
                val_ratio=train_cfg.get("val_ratio", 0.2),
                seed=train_cfg.get("seed", 42),
            )

            train_df = splits["train"]
            val_df = splits["val"]

            target_col = data.target_col
            id_col = data.id_col

            sample_weight_col = "sample_weight"
            train_sample_weight = None
            val_sample_weight = None

            drop_cols = [target_col]
            if id_col and id_col in train_df.columns:
                drop_cols.append(id_col)
            if sample_weight_col in train_df.columns:
                train_sample_weight = pd.to_numeric(
                    train_df[sample_weight_col], errors="coerce",
                ).fillna(1.0).values
                drop_cols.append(sample_weight_col)
            if sample_weight_col in val_df.columns:
                val_sample_weight = pd.to_numeric(
                    val_df[sample_weight_col], errors="coerce",
                ).fillna(1.0).values
                if sample_weight_col not in drop_cols:
                    drop_cols.append(sample_weight_col)

            X_train = train_df.drop(columns=drop_cols, errors="ignore")
            y_train = train_df[target_col].values
            X_val = val_df.drop(columns=drop_cols, errors="ignore")
            y_val = val_df[target_col].values

            result["train_shape"] = X_train.shape
            result["n_features"] = X_train.shape[1]

            cat_cols = [c for c in X_train.columns if X_train[c].dtype.kind in ("O", "b")]

            # ------------------------------------------------------------------
            # Model dispatch: LightGBM (default) | LSTM/GRU/Transformer (seq).
            # Mirrors the routing in scripts/train.py so every tabular task in
            # ``DATASET_REGISTRY`` is supported, not just LightGBM ones.
            # ------------------------------------------------------------------
            if model_name in SEQUENCE_MODELS:
                from dppbench.models import (
                    LSTMForecaster, GRUForecaster, TransformerForecaster,
                )
                MODEL_MAP = {
                    "LSTM": LSTMForecaster,
                    "GRU": GRUForecaster,
                    "Transformer": TransformerForecaster,
                }
                X_train_num = X_train.copy()
                X_val_num = X_val.copy()
                for c in cat_cols:
                    X_train_num[c] = pd.to_numeric(X_train_num[c], errors="coerce")
                    X_val_num[c] = pd.to_numeric(X_val_num[c], errors="coerce")
                X_train_num = X_train_num.fillna(0.0)
                X_val_num = X_val_num.fillna(0.0)
                model_params = {**model_params, "device": self.device}
                model = MODEL_MAP[model_name](**model_params)
                model.fit(X_train_num, y_train, X_val=X_val_num, y_val=y_val)
                metrics = train_cfg.get(
                    "metrics", ["rmse"] if model_params.get("task") != "binary" else ["auc"],
                )
                val_result = model.evaluate(X_val_num, y_val, metrics=metrics)
                eval_X_train_for_std = X_train  # column reference only
                eval_use_seq = True
            elif model_name in TABULAR_DEEP_MODELS:
                MODEL_MAP = {
                    "MLP": MLP,
                    "TabTransformer": TabTransformer,
                    "FTTransformer": FTTransformer,
                    "SAINT": SAINT,
                }
                model_params = {**model_params, "device": self.device}
                model = MODEL_MAP[model_name](**model_params)
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                metrics = train_cfg.get(
                    "metrics", ["auc"] if model_params.get("task", "binary") == "binary" else ["rmse"],
                )
                val_result = model.evaluate(X_val, y_val, metrics=metrics)
                eval_X_train_for_std = X_train
                eval_use_seq = False
            else:
                if model_name != "LightGBM":
                    raise ValueError(
                        f"Unsupported tabular model '{model_name}' for {self.data_name}."
                    )
                model = LightGBMModel(**model_params)
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    sample_weight=train_sample_weight,
                    eval_sample_weight=[val_sample_weight] if val_sample_weight is not None else None,
                    categorical_features=cat_cols or "auto",
                )
                metrics = train_cfg.get(
                    "metrics", ["auc"] if model_params.get("task", "binary") == "binary" else ["rmse"],
                )
                val_result = model.evaluate(X_val, y_val, metrics=metrics)
                eval_X_train_for_std = X_train
                eval_use_seq = False

            result["success"] = True
            result["metrics"] = dict(val_result)

            std_test_df = splits.get("std_test")
            if std_test_df is not None and len(std_test_df) > 0 and target_col in std_test_df.columns:
                X_std = std_test_df.drop(columns=drop_cols, errors="ignore")
                missing = [c for c in eval_X_train_for_std.columns if c not in X_std.columns]
                for c in missing:
                    X_std[c] = 0
                X_std = X_std[eval_X_train_for_std.columns]
                y_std = std_test_df[target_col].values
                if eval_use_seq:
                    X_std_eval = X_std.copy()
                    for c in cat_cols:
                        if c in X_std_eval.columns:
                            X_std_eval[c] = pd.to_numeric(X_std_eval[c], errors="coerce")
                    X_std_eval = X_std_eval.fillna(0.0)
                else:
                    X_std_eval = X_std
                _t_inf = time.time()
                std_result = model.evaluate(X_std_eval, y_std, metrics=metrics)
                result["metrics"]["std_test_inference_seconds"] = round(
                    time.time() - _t_inf, 4
                )
                result["metrics"]["std_test_n_rows"] = int(len(std_test_df))
                for k, v in std_result.items():
                    result["metrics"][f"std_test_{k}"] = v

            try:
                importance = model.feature_importance()
                result["top_features"] = list(importance.items())[:15]
            except (NotImplementedError, AttributeError):
                result["top_features"] = []

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"

        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result

    def _run_graph_training(self):
        """Train a GNN (GCN / GraphSAGE / GAT) on a node-classification task.

        Std-test evaluation for graph tasks is left as a no-op for now: the
        std-test rows live in ``data.train_df`` (tagged with
        ``__split__='std_test'``) but the GNN training path consumes a single
        full-graph object via ``data.build_graph()``, which already routes
        labels through its own train/val/test masks. Adding a separate
        std-test mask requires upstream support that is not in scope here.
        """
        from dppbench.models import train_graph, GCN, GraphSAGE, GAT

        start_time = time.time()
        result = {
            "success": False,
            "metrics": {},
            "error": None,
            "train_shape": None,
            "n_features": None,
            "duration_seconds": None,
        }

        try:
            data = self._make_configured_data_instance()
            data.load_data()
            data.run_pre_process(os.path.abspath(self._working_yaml))

            cfg = self.get_model_config()
            model_name = cfg.get("model", "GraphSAGE")
            self._apply_fast_overrides(cfg, model_name)
            train_cfg = cfg.get("train", {})
            model_params = dict(cfg.get("model_params", {}))

            if not hasattr(data, "build_graph"):
                raise ValueError(
                    f"Dataset '{self.data_name}' does not implement build_graph(); "
                    f"cannot train GNN model '{model_name}'."
                )

            graph = data.build_graph()
            MODEL_MAP = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT}
            if model_name not in MODEL_MAP:
                raise ValueError(
                    f"Unsupported graph model '{model_name}' for {self.data_name}; "
                    f"expected one of {sorted(MODEL_MAP)}."
                )
            ctor_kwargs = {
                k: v for k, v in model_params.items()
                if k not in ("task", "epochs", "lr", "weight_decay", "seed")
            }
            model = MODEL_MAP[model_name](in_dim=graph["x"].shape[1], **ctor_kwargs)
            graph_result = train_graph(model, graph, model_params, train_cfg, device=self.device)

            result["success"] = True
            result["metrics"] = dict(graph_result) if isinstance(graph_result, dict) else {"result": graph_result}
            result["train_shape"] = (int(graph["x"].shape[0]), int(graph["x"].shape[1]))
            result["n_features"] = int(graph["x"].shape[1])

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"

        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result

    def _run_rec_training(self):
        from dppbench.utils import get_model
        from dppbench.models import build_feature_columns, df_to_input

        start_time = time.time()
        result = {
            "success": False,
            "metrics": {},
            "error": None,
            "train_shape": None,
            "n_features": None,
            "duration_seconds": None,
        }

        try:
            data = self._make_configured_data_instance()
            cfg = self.get_model_config()
            data.load_data()
            data.run_pre_process(os.path.abspath(self._working_yaml))
            splits = data.split()

            cfg.setdefault("model_params", {})["device"] = self.device
            self._apply_fast_overrides(cfg, cfg.get("model", "DIN"))

            all_df = pd.concat(list(splits.values()), ignore_index=True)
            feature_columns = build_feature_columns(all_df, cfg, col_types=data.col_types)

            model = get_model(feature_columns, cfg)

            train_cfg = cfg.get("train", {})
            history, test_results = model.train_and_evaluate(
                splits["train"], splits["test"],
                feature_columns, cfg
            )

            result["success"] = True
            result["metrics"] = dict(test_results)

            std_test_df = splits.get("test")
            if std_test_df is not None and len(std_test_df) > 0:
                try:
                    std_x, std_y = df_to_input(std_test_df, feature_columns, cfg)
                    _t_inf = time.time()
                    std_result = model.evaluate(
                        std_x, std_y,
                        batch_size=train_cfg.get("batch_size", 256),
                    )
                    result["metrics"]["std_test_inference_seconds"] = round(
                        time.time() - _t_inf, 4
                    )
                    result["metrics"]["std_test_n_rows"] = int(len(std_test_df))
                    for k, v in std_result.items():
                        result["metrics"][f"std_test_{k}"] = v
                except Exception as eval_err:
                    print(f"  [std_test] eval failed: {eval_err}")

            result["train_shape"] = (len(splits["train"]), len(feature_columns))
            result["n_features"] = len(feature_columns)

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"

        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result
