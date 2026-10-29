import os
import sys
import yaml
import time
import tempfile
import traceback
import shutil
import copy
import gc
import hashlib
import json
import pandas as pd

from baselines.common.data_loader import DATASET_REGISTRY, TaskDataLoader

# Repo root (= parent of baselines/) so ``import dppbench`` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


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
                 model_name=None, model_config=None, seed=None):
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.device = device
        self.model_name = model_name
        self.model_config = model_config
        self.seed = int(seed) if seed is not None else None
        self.original_yaml = os.path.join(task_dir, "prepare.yaml")
        self.model_yaml = os.path.join(task_dir, "model.yaml")
        if data_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {data_name}. Available: {list(DATASET_REGISTRY.keys())}")
        self._data_loader = TaskDataLoader(
            data_name,
            data_dir=data_dir,
            model_config_provider=self.get_model_config,
        )
        self._task_type = self._data_loader.task_type

        self._work_dir = tempfile.mkdtemp(prefix="preproc_opt_")
        self._working_yaml = os.path.join(self._work_dir, "prepare.yaml")
        shutil.copy2(self.original_yaml, self._working_yaml)

        # Search-time fast mode toggle. When enabled, training rounds
        # (n_estimators / epochs) are reduced via _apply_fast_overrides at the
        # start of each ``run_training()`` call.
        self._fast_mode = False
        self._fast_overrides = None
        self._evaluate_std_test = True

    def set_evaluate_std_test(self, enabled: bool):
        """Enable/disable frozen std-test evaluation for this executor."""
        self._evaluate_std_test = bool(enabled)

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

    def _resolved_data_dir(self):
        return self._data_loader.resolved_data_dir()

    def _make_data_instance(self):
        return self._data_loader.make_data_instance()

    def _make_configured_data_instance(self):
        data = self._make_data_instance()
        if self.task_type == "graph" and not self._evaluate_std_test:
            original_load = data.load_data
            def load_graph_search_data(*args, **kwargs):
                original_load(*args, **kwargs)
                path = os.path.join(data._std_test_dir(data.data_dir), "train_frozen.parquet")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Graph search requires frozen training data: {path}")
                data.train_df = data._normalize_frozen_frame(pd.read_parquet(path))
                if "__split__" in data.train_df and not data.train_df["__split__"].eq("train").all():
                    raise ValueError("Graph search frozen training file contains held-out split markers")
                data._skip_std_test_loading = True
                small_n = getattr(data, "_search_small_n", 0)
                if small_n:
                    from .search_evaluator import subsample_data_inplace
                    subsample_data_inplace(data, small_n, getattr(data, "_search_seed", 42))
                return data.train_df, data.test_df
            data.load_data = load_graph_search_data
        return self._data_loader.configure_data_instance(data)

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
            output_path = os.path.join(self._work_dir, "best_prepare.yaml")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def cleanup(self):
        self.clear_data_cache()
        if os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def get_model_config(self):
        with open(self.model_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = self._resolve_model_config(cfg)
        if self.seed is not None:
            cfg.setdefault("model_params", {})["seed"] = self.seed
            cfg.setdefault("train", {})["seed"] = self.seed
        return cfg

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

    def _load_data(self):
        return self.load_data()

    def load_data(self):
        if self._data_loader._data is None:
            self._data_loader._data = self._make_configured_data_instance()
            self._data_loader._data.load_data()
        return self._data_loader._data

    def clear_data_cache(self):
        self._data_loader.clear_cache()

    def run_training(self):
        # Search sampling wraps bound load methods in closures. Collect the
        # previous candidate's unreachable data/model cycles before loading
        # another full dataset (also applies to final HPO trials).
        gc.collect()
        if self._task_type == "tabular":
            return self._run_tabular_training()
        elif self._task_type == "graph":
            return self._run_graph_training()
        else:
            return self._run_rec_training()

    def run_preprocessing(self):
        """Execute only the working preparation DAG and return its primary frame.

        CtxPipe uses this to mirror the original environment, where every
        selected primitive is applied before the next state/context is built.
        """
        data = self._make_configured_data_instance()
        data.load_data()
        data.run_pre_process(os.path.abspath(self._working_yaml))
        frame = self._primary_frame(data)
        return frame.copy(deep=True) if frame is not None else None

    def run_static_quality(self):
        """Score a pipeline with cheap post-preprocess data-quality proxies.

        This is a search-time objective: it never uses std-test labels for the
        returned fitness. If std-test rows are present because the shared
        preprocessing loader appends them, they are dropped before profiling.
        """
        return self._run_quality_score(method="static_quality")

    def run_error_detection_quality(self):
        """Score whether a pipeline targets actual detected data issues.

        Instead of injecting synthetic noise, compare raw-vs-processed issue
        profiles and reward operators that cover problems actually present in
        the raw task data. This keeps the proxy deterministic and avoids
        rewarding arbitrary destructive transforms.
        """
        return self._run_quality_score(method="error_detection")

    def _run_quality_score(self, method):
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
            raw_df = self._drop_std_test_rows(self._primary_frame(data))
            raw_profile = self._quality_profile(raw_df, self._target_col_for_quality(data))
            raw_rows = int(len(raw_df)) if raw_df is not None else 0

            data.run_pre_process(os.path.abspath(self._working_yaml))
            result["execution_data"] = self._execution_profile(data)
            processed_df = self._drop_std_test_rows(self._primary_frame(data))
            target_col = self._target_col_for_quality(data)
            post_profile = self._quality_profile(processed_df, target_col)

            if processed_df is None or len(processed_df) == 0:
                raise ValueError("pipeline produced no rows for quality scoring")

            if method == "static_quality":
                metrics = self._static_quality_metrics(
                    raw_profile=raw_profile,
                    post_profile=post_profile,
                    processed_df=processed_df,
                    raw_rows=raw_rows,
                )
            elif method == "error_detection":
                op_names = self._current_pipeline_ops()
                metrics = self._error_detection_metrics(
                    raw_profile=raw_profile,
                    post_profile=post_profile,
                    op_names=op_names,
                )
            else:
                raise ValueError(f"unknown quality scoring method: {method}")

            result["success"] = True
            result["metrics"] = metrics
            result["train_shape"] = processed_df.shape
            result["n_features"] = int(max(0, processed_df.shape[1] - (1 if target_col in processed_df.columns else 0)))
        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result

    def _primary_frame(self, data):
        if self._task_type == "rec":
            return getattr(data, "interaction_df", None)
        return getattr(data, "train_df", None)

    def _execution_profile(self, data):
        """Return the post-pipeline primary-frame schema for legality checks."""
        frame = self._primary_frame(data)
        if frame is None:
            return None
        return {
            "shape": [int(frame.shape[0]), int(frame.shape[1])],
            "columns": [str(col) for col in frame.columns],
            "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        }

    @staticmethod
    def _drop_std_test_rows(df):
        if df is None:
            return None
        if "__split__" in df.columns:
            return df[df["__split__"] != "std_test"].drop(columns="__split__").reset_index(drop=True)
        return df.reset_index(drop=True)

    def _target_col_for_quality(self, data):
        if self._task_type == "rec":
            cfg = getattr(data, "model_cfg", {}) or self.get_model_config()
            return (cfg.get("feature", {}) or {}).get("target_col", "rating")
        return getattr(data, "target_col", None)

    def _current_pipeline_ops(self):
        with open(self._working_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        dag = cfg.get("dag") if isinstance(cfg, dict) else None
        if not isinstance(dag, dict):
            return []
        return [node.get("op") for node in (dag.get("ops") or []) if isinstance(node, dict)]

    @staticmethod
    def _safe_ratio(num, den):
        return float(num) / float(den) if den else 0.0

    def _quality_profile(self, df, target_col=None):
        import numpy as np

        if df is None or len(df) == 0:
            return {
                "n_rows": 0,
                "n_cols": 0,
                "missing_cell_ratio": 1.0,
                "duplicate_row_ratio": 1.0,
                "constant_col_ratio": 1.0,
                "numeric_outlier_ratio": 0.0,
                "categorical_rare_ratio": 0.0,
                "label_imbalance_penalty": 0.0,
            }
        work = df.drop(columns=["__split__"], errors="ignore")
        map_values = getattr(work, "map", None)
        hashable_work = (
            map_values(self._hashable_quality_value)
            if callable(map_values)
            else work.applymap(self._hashable_quality_value)
        )
        n_rows, n_cols = work.shape
        missing_cell_ratio = float(work.isna().mean().mean()) if n_cols else 0.0
        duplicate_row_ratio = self._safe_ratio(int(hashable_work.duplicated().sum()), n_rows)

        feature_cols = [c for c in work.columns if c != target_col]
        if feature_cols:
            nunique = hashable_work[feature_cols].nunique(dropna=True)
            constant_col_ratio = self._safe_ratio(int((nunique <= 1).sum()), len(feature_cols))
        else:
            constant_col_ratio = 1.0

        numeric = work[feature_cols].select_dtypes(include="number") if feature_cols else work.iloc[:, 0:0]
        outlier_cells = 0
        numeric_cells = 0
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) < 8:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_cells += int(((s < lo) | (s > hi)).sum())
            numeric_cells += int(len(s))
        numeric_outlier_ratio = self._safe_ratio(outlier_cells, numeric_cells)

        cat = work[feature_cols].select_dtypes(exclude="number") if feature_cols else work.iloc[:, 0:0]
        rare_cells = 0
        cat_cells = 0
        for col in cat.columns:
            s = hashable_work[col].dropna()
            if len(s) < 10:
                continue
            counts = s.value_counts(normalize=True)
            rare_values = set(counts[counts < 0.01].index)
            rare_cells += int(s.isin(rare_values).sum())
            cat_cells += int(len(s))
        categorical_rare_ratio = self._safe_ratio(rare_cells, cat_cells)

        label_imbalance_penalty = 0.0
        if target_col and target_col in work.columns:
            y = work[target_col].dropna()
            if len(y) > 0:
                freq = y.value_counts(normalize=True)
                if len(freq) > 1:
                    label_imbalance_penalty = float(max(0.0, freq.max() - (1.0 / len(freq))))

        return {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "missing_cell_ratio": float(np.clip(missing_cell_ratio, 0.0, 1.0)),
            "duplicate_row_ratio": float(np.clip(duplicate_row_ratio, 0.0, 1.0)),
            "constant_col_ratio": float(np.clip(constant_col_ratio, 0.0, 1.0)),
            "numeric_outlier_ratio": float(np.clip(numeric_outlier_ratio, 0.0, 1.0)),
            "categorical_rare_ratio": float(np.clip(categorical_rare_ratio, 0.0, 1.0)),
            "label_imbalance_penalty": float(np.clip(label_imbalance_penalty, 0.0, 1.0)),
        }

    @staticmethod
    def _hashable_quality_value(value):
        import numpy as np

        if isinstance(value, np.ndarray):
            return tuple(TrainingExecutor._hashable_quality_value(v) for v in value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(TrainingExecutor._hashable_quality_value(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(TrainingExecutor._hashable_quality_value(v) for v in value))
        if isinstance(value, dict):
            return tuple(sorted(
                (TrainingExecutor._hashable_quality_value(k), TrainingExecutor._hashable_quality_value(v))
                for k, v in value.items()
            ))
        return value

    @staticmethod
    def _clip01(value):
        return max(0.0, min(1.0, float(value)))

    def _static_quality_metrics(self, raw_profile, post_profile, processed_df, raw_rows):
        retention = self._safe_ratio(post_profile["n_rows"], raw_rows) if raw_rows else 1.0
        retention = self._clip01(retention)
        width_penalty = 0.0
        if post_profile["n_rows"] > 0:
            width_penalty = self._clip01(max(0, post_profile["n_cols"] - 500) / 500.0)

        feature_penalty = (
            0.35 * post_profile["missing_cell_ratio"]
            + 0.20 * post_profile["constant_col_ratio"]
            + 0.20 * post_profile["numeric_outlier_ratio"]
            + 0.15 * post_profile["categorical_rare_ratio"]
            + 0.10 * width_penalty
        )
        sample_penalty = (
            0.50 * post_profile["duplicate_row_ratio"]
            + 0.30 * (1.0 - retention)
            + 0.20 * post_profile["label_imbalance_penalty"]
        )
        feature_quality = self._clip01(1.0 - feature_penalty)
        sample_quality = self._clip01(1.0 - sample_penalty)
        static_quality = self._clip01(0.6 * feature_quality + 0.4 * sample_quality)
        return {
            "static_quality": static_quality,
            "feature_quality": feature_quality,
            "sample_quality": sample_quality,
            "sample_retention": retention,
            "raw_rows": raw_profile["n_rows"],
            **{f"post_{k}": v for k, v in post_profile.items()},
        }

    def _error_detection_metrics(self, raw_profile, post_profile, op_names):
        issue_map = {
            "missing": {
                "severity": raw_profile["missing_cell_ratio"],
                "post": post_profile["missing_cell_ratio"],
                "ops": {"HandleMV", "CustomClean"},
            },
            "duplicate": {
                "severity": raw_profile["duplicate_row_ratio"],
                "post": post_profile["duplicate_row_ratio"],
                "ops": {"Deduplicate"},
            },
            "outlier": {
                "severity": raw_profile["numeric_outlier_ratio"],
                "post": post_profile["numeric_outlier_ratio"],
                "ops": {"HandleOutlier", "ClipOutlier", "HandleError"},
            },
            "rare_category": {
                "severity": raw_profile["categorical_rare_ratio"],
                "post": post_profile["categorical_rare_ratio"],
                "ops": {"FrequencyEncode", "HashEncode", "OneHotEncode", "TargetEncode"},
            },
            "imbalance": {
                "severity": raw_profile["label_imbalance_penalty"],
                "post": post_profile["label_imbalance_penalty"],
                "ops": {"Oversample", "Undersample", "SampleNegative"},
            },
        }
        op_set = set(op_names)
        total_severity = sum(v["severity"] for v in issue_map.values())
        weighted_coverage = 0.0
        weighted_repair = 0.0
        unnecessary = 0
        relevant = 0
        for spec in issue_map.values():
            severity = spec["severity"]
            handles = bool(op_set & spec["ops"])
            if severity > 0.01:
                relevant += 1
                if handles:
                    weighted_coverage += severity
                gain = max(0.0, severity - spec["post"])
                weighted_repair += gain
            elif handles:
                unnecessary += 1
        denom = total_severity if total_severity > 1e-12 else 1.0
        coverage_score = self._clip01(weighted_coverage / denom)
        repair_score = self._clip01(weighted_repair / denom)
        risk_penalty = min(0.3, 0.06 * unnecessary)
        error_detection = self._clip01(0.55 * repair_score + 0.35 * coverage_score + 0.10 - risk_penalty)
        return {
            "error_detection": error_detection,
            "repair_score": repair_score,
            "coverage_score": coverage_score,
            "risk_penalty": risk_penalty,
            "relevant_issue_count": relevant,
            "unnecessary_operator_count": unnecessary,
            "ops": list(op_names),
            **{f"raw_{k}": v for k, v in raw_profile.items()},
            **{f"post_{k}": v for k, v in post_profile.items()},
        }

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
            result["execution_data"] = self._execution_profile(data)

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
            print(f"[model inputs] columns={list(X_train.columns)}")
            print(f"[model inputs] train_sha256={_frame_digest(X_train)} val_sha256={_frame_digest(X_val)}")

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
            if self._evaluate_std_test and std_test_df is not None and len(std_test_df) > 0 and target_col in std_test_df.columns:
                X_std = std_test_df.drop(columns=drop_cols, errors="ignore")
                missing = [c for c in eval_X_train_for_std.columns if c not in X_std.columns]
                for c in missing:
                    X_std[c] = 0
                X_std = X_std[eval_X_train_for_std.columns]
                y_std = std_test_df[target_col].values
                print(f"[model inputs] std_test_sha256={_frame_digest(X_std)}")
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
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

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
            if self._evaluate_std_test:
                data.load_std_test_frozen()
            data.run_pre_process(os.path.abspath(self._working_yaml))
            result["execution_data"] = self._execution_profile(data)

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
            graph_result = train_graph(
                model, graph, model_params, train_cfg, device=self.device,
                evaluate_test=self._evaluate_std_test,
            )

            result["success"] = True
            raw_metrics = (
                dict(graph_result)
                if isinstance(graph_result, dict)
                else {"result": graph_result}
            )
            val_metrics = {
                key[len("val_"):]: value
                for key, value in raw_metrics.items()
                if key.startswith("val_")
            }
            result["metrics"] = val_metrics or raw_metrics
            if self._evaluate_std_test and val_metrics:
                for key, value in raw_metrics.items():
                    if key.startswith("std_test_"):
                        result["metrics"][key] = value
                        if key == "std_test_n_nodes":
                            result["metrics"]["std_test_n_rows"] = value
                    elif not key.startswith("val_"):
                        result["metrics"][f"test_{key}"] = value
            result["train_shape"] = (int(graph["x"].shape[0]), int(graph["x"].shape[1]))
            result["n_features"] = int(graph["x"].shape[1])

        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result

    def _run_rec_training(self):
        from baselines.common.budget import memory_checkpoint
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
            memory_checkpoint("rec.before_load")
            data.load_data()
            memory_checkpoint("rec.loaded")
            data.run_pre_process(os.path.abspath(self._working_yaml))
            memory_checkpoint("rec.preprocessed")
            result["execution_data"] = self._execution_profile(data)
            splits = data.split()

            cfg.setdefault("model_params", {})["device"] = self.device
            self._apply_fast_overrides(cfg, cfg.get("model", "DIN"))

            frames_for_features = [splits["train"]]
            if self._evaluate_std_test and "test" in splits:
                frames_for_features.append(splits["test"])
            if len(frames_for_features) > 1:
                from dppbench.memory import require_memory_headroom
                require_memory_headroom(sum(int(frame.memory_usage(index=False, deep=False).sum())
                                            for frame in frames_for_features), "rec.feature_schema")
            all_df = (frames_for_features[0] if len(frames_for_features) == 1
                      else pd.concat(frames_for_features, ignore_index=True))
            feature_columns = build_feature_columns(all_df, cfg, col_types=data.col_types)
            used_features = [fc.name for fc in feature_columns]
            ignored = [c for c in all_df.columns if c not in used_features and c not in {"__split__", cfg["feature"]["target_col"]}]
            print(f"[model inputs] features={used_features} ignored={ignored}")

            del all_df, frames_for_features
            model = get_model(feature_columns, cfg)
            memory_checkpoint("rec.model_built")
            if os.environ.get("DPPBENCH_MEMORY_TRACE") == "1":
                print(f"[memory] model_parameters={sum(p.numel() for p in model.parameters())}", flush=True)

            train_cfg = cfg.get("train", {})
            history, test_results = model.train_and_evaluate(
                splits["train"], splits["test"],
                feature_columns, cfg,
                evaluate_test=self._evaluate_std_test,
            )
            memory_checkpoint("rec.trained")

            result["success"] = True
            result["metrics"] = self._rec_validation_metrics(history, train_cfg)
            if not result["metrics"]:
                raise ValueError("Missing validation metrics; test metrics cannot replace validation fitness")

            std_test_df = splits.get("test")
            if self._evaluate_std_test and std_test_df is not None and len(std_test_df) > 0:
                result["metrics"]["std_test_n_rows"] = int(len(std_test_df))
                for k, v in (test_results or {}).items():
                    result["metrics"][f"std_test_{k}"] = v
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
                    for k, v in std_result.items():
                        result["metrics"][f"std_test_{k}"] = v
                except Exception as eval_err:
                    raise RuntimeError(f"Standard-test evaluation failed: {eval_err}") from eval_err

            result["train_shape"] = (len(splits["train"]), len(feature_columns))
            result["n_features"] = len(feature_columns)

        except Exception as e:
            result["success"] = False
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        result["duration_seconds"] = round(time.time() - start_time, 1)
        return result

    @staticmethod
    def _rec_validation_metrics(history, train_cfg):
        if not history:
            return {}
        metrics = {}
        higher_is_better = {
            "auc": True,
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True,
            "ndcg": True,
            "logloss": False,
            "mse": False,
            "rmse": False,
            "mae": False,
        }
        # Report all metrics from the actual retained checkpoint, not a
        # different independently optimal epoch for every metric.
        epoch = -1
        monitor = train_cfg.get("early_stopping_monitor")
        if monitor and train_cfg.get("early_stopping_patience", 0) > 0:
            values = history.get(f"val_{monitor}", [])
            candidates = [(i, float(v)) for i, v in enumerate(values) if pd.notna(v)]
            if candidates:
                choose = max if higher_is_better.get(monitor, True) else min
                epoch = choose(candidates, key=lambda pair: pair[1])[0]
        for name in train_cfg.get("metrics", ["auc"]):
            values = history.get(f"val_{name}", [])
            if not values:
                continue
            metrics[name] = float(values[epoch])
        return metrics


def _frame_digest(frame):
    """Include schema and row order in the fingerprint of tabular model input."""
    digest = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    try:
        values = pd.util.hash_pandas_object(frame, index=False)
    except TypeError:
        values = pd.util.hash_pandas_object(frame.astype(str), index=False)
    digest.update(values.values.tobytes())
    return digest.hexdigest()
