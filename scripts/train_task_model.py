import sys
import os
import argparse
import json
import copy
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd
from scripts.build_std_test import REC_SPLIT_METHOD, rec_cold_start_user_filter

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dppbench", "tasks")

REC_DATASETS = {"amazon_beauty", "kuairec", "movielens", "yelp", "tenrec"}
REC_MODELS = {"FNN", "DeepFM", "DIN", "DIEN", "SIM"}
TABULAR_DEEP_MODELS = {"MLP", "TabTransformer", "FTTransformer", "SAINT"}
SEQUENCE_MODELS = {"LSTM", "GRU", "Transformer"}
GRAPH_MODELS = {"GCN", "GraphSAGE", "GAT"}

TABULAR_REGISTRY = {
    "home_credit": "dppbench.tasks.home_credit.home_credit_data.HomeCreditData",
    "fraud_detection": "dppbench.tasks.fraud_detection.fraud_detection_data.FraudDetectionData",
    "berka": "dppbench.tasks.berka.berka_data.BerkaData",
    "bondora": "dppbench.tasks.bondora.bondora_data.BondoraData",
    "default_credit": "dppbench.tasks.default_credit.default_credit_data.DefaultCreditData",
    "polish_bankruptcy": "dppbench.tasks.polish_bankruptcy.polish_bankruptcy_data.PolishBankruptcyData",
    "bike_sharing": "dppbench.tasks.bike_sharing.bike_sharing_data.BikeSharingData",
    "beijing_air_quality": "dppbench.tasks.beijing_air_quality.beijing_air_quality_data.BeijingAirQualityData",
    "nyc_taxi_hourly": "dppbench.tasks.nyc_taxi_hourly.nyc_taxi_hourly_data.NycTaxiHourlyData",
    "citibike_jc_hourly": "dppbench.tasks.citibike_jc_hourly.citibike_jc_hourly_data.CitibikeJcHourlyData",
    "elliptic_bitcoin": "dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data.EllipticBitcoinData",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="amazon_beauty")
    parser.add_argument("--model", type=str, default=None,
                         help="Override the model name from model.yaml/model_options.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/sdb/dengjiale/dppbench/data",
        help=(
            "Optional dataset root. When set, task files are stored under "
            "<data_dir>/<data_name>/data and std_test under "
            "<data_dir>/<data_name>/std_test."
        ),
    )
    parser.add_argument(
        "--gpu_id", type=int, default=-1,
        help="GPU index to use (-1 = CPU). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/sdb/dengjiale/dppbench/output",
        help=(
            "Optional output root. When set, logs and structured metrics are "
            "written under <output_dir>/baseline/<data_name>/<model>."
        ),
    )
    return parser.parse_args()


def _resolve_device(gpu_id):
    if gpu_id is None or gpu_id < 0:
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            print(f"[warn] gpu_id={gpu_id} requested but CUDA not available; falling back to CPU.")
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda:0"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _output_paths(output_dir, data_name, model_name):
    if not output_dir:
        return None
    run_dir = os.path.join(
        os.path.abspath(output_dir), "baseline", data_name, model_name
    )
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log": os.path.join(run_dir, "stdout.log"),
        "metrics": os.path.join(run_dir, "metrics.json"),
        "config": os.path.join(run_dir, "run_config.json"),
    }


def _parse_metrics_from_log(log_path):
    metrics = {}
    if not log_path or not os.path.exists(log_path):
        return metrics
    prefixes = {"val:": "val", "std_test:": "std_test", "test:": "test"}
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            for prefix, key in prefixes.items():
                if line.startswith(prefix):
                    payload = line[len(prefix):].strip()
                    try:
                        metrics[key] = ast.literal_eval(payload)
                    except (SyntaxError, ValueError):
                        metrics[key] = {"raw": payload}
                    break
    return metrics


def _write_run_artifacts(paths, args, cfg, model_name, device, error=None):
    if not paths:
        return
    run_config = {
        "data_name": args.data_name,
        "model": model_name,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "gpu_id": args.gpu_id,
        "device": device,
        "model_config": cfg,
    }
    if error is not None:
        run_config["error"] = error
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    metrics = _parse_metrics_from_log(paths["log"])
    if error is not None:
        metrics["error"] = error
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _resolve_model_config(cfg, model_name=None):
    cfg = copy.deepcopy(cfg)
    if not model_name:
        return cfg
    options = cfg.get("model_options") or {}
    if model_name not in options:
        raise ValueError(
            f"Model '{model_name}' is not configured. Available: "
            f"{sorted(options) or [cfg.get('model')]}"
        )
    option = options[model_name] or {}
    cfg["model"] = model_name
    for key, value in option.items():
        if key in {"model_params", "train", "feature"} and isinstance(value, dict):
            cfg.setdefault(key, {})
            cfg[key].update(copy.deepcopy(value))
        elif key != "model":
            cfg[key] = copy.deepcopy(value)
    return cfg


def _task_data_dir(data_name, data_dir=None):
    if not data_dir:
        return None
    return os.path.join(os.path.abspath(data_dir), data_name, "data")


def _std_test_dir(data_name, data_dir=None):
    task_data_dir = _task_data_dir(data_name, data_dir)
    if task_data_dir:
        return os.path.join(os.path.dirname(task_data_dir), "std_test")
    return os.path.abspath(os.path.join(BASE_DIR, data_name, "std_test"))


def _required_std_test_files(data_name):
    if data_name in REC_DATASETS:
        return ("std_test.parquet", "interaction_frozen.parquet")
    return ("std_test.parquet", "train_frozen.parquet")


def _load_std_test_meta(std_dir):
    meta_path = os.path.join(std_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_std_test(args, cfg):
    std_dir = _std_test_dir(args.data_name, args.data_dir)
    required_files = _required_std_test_files(args.data_name)
    missing = [
        filename for filename in required_files
        if not os.path.exists(os.path.join(std_dir, filename))
    ]
    rebuild_reason = None
    if missing:
        rebuild_reason = f"missing {missing} under {std_dir}"
    elif args.data_name in REC_DATASETS:
        meta = _load_std_test_meta(std_dir)
        expected_rule = cfg.get("feature", {}).get("label_rule")
        actual_rule = meta.get("label_rule")
        if actual_rule != expected_rule:
            rebuild_reason = (
                f"label_rule changed for {args.data_name}; "
                "rebuilding standard test data"
            )
        elif meta.get("split_method") != REC_SPLIT_METHOD:
            rebuild_reason = (
                f"std-test split protocol changed for {args.data_name}; "
                "rebuilding standard test data"
            )
        elif (
            meta.get("cold_start_user_filter")
            != rec_cold_start_user_filter(args.data_name, cfg)
        ):
            rebuild_reason = (
                f"std-test cold-start user filter changed for {args.data_name}; "
                "rebuilding standard test data"
            )

    if rebuild_reason is None:
        print(f"[std_test] using standard test data from {std_dir}")
        return

    print(f"[std_test] {rebuild_reason}; building standard test data first...")
    from scripts.build_std_test import run_for_task
    run_for_task(args.data_name, dry_run=False, data_dir=args.data_dir)


def _train_rec(args, cfg, pre_process_yaml, device="cpu"):
    from dppbench.utils import get_data, get_model
    data = get_data(
        args.data_name,
        _task_data_dir(args.data_name, args.data_dir),
        pre_process_yaml,
        cfg,
    )
    splits, feature_columns = data
    cfg.setdefault("model_params", {})["device"] = device
    model = get_model(feature_columns, cfg)
    print("-" * 60)
    print("Training...")
    history, test_results = model.train_and_evaluate(
        splits["train"], splits["test"], feature_columns, cfg
    )
    print(f"test: {test_results}")


def _load_tabular_class(data_name):
    import importlib
    if data_name not in TABULAR_REGISTRY:
        raise ValueError(
            f"Unknown tabular dataset '{data_name}'. "
            f"Available: {sorted(TABULAR_REGISTRY)}"
        )
    module_path, class_name = TABULAR_REGISTRY[data_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _train_tabular(args, cfg, pre_process_yaml, device="cpu"):
    from dppbench.models import (
        LightGBMModel, MLP, TabTransformer, FTTransformer, SAINT,
    )

    print("=" * 60)
    print(f"Dataset: {args.data_name}")
    print("=" * 60)

    data_cls = _load_tabular_class(args.data_name)
    data = data_cls(data_dir=_task_data_dir(args.data_name, args.data_dir))
    data.load_data()
    print(f"Loaded train: {data.train_df.shape}, "
          f"test: {data.test_df.shape if data.test_df is not None else None}")
    print("-" * 60)
    print("Preprocessing...")
    data.run_pre_process(os.path.abspath(pre_process_yaml))

    train_cfg = cfg.get("train", {})
    model_name = cfg.get("model", "LightGBM")
    model_params = cfg.get("model_params", {})

    # GNN branch (graph datasets): bypass tabular split / matrix flow.
    if model_name in GRAPH_MODELS:
        from dppbench.models import train_graph, GCN, GraphSAGE, GAT
        MODEL_MAP = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT}
        if not hasattr(data, "build_graph"):
            raise ValueError(
                f"Dataset '{args.data_name}' does not implement build_graph(); "
                f"cannot train GNN model '{model_name}'."
            )
        graph = data.build_graph()
        ctor_kwargs = {k: v for k, v in model_params.items()
                       if k not in ("task", "epochs", "lr", "weight_decay", "seed")}
        print("-" * 60)
        print(f"Training {model_name}...")
        model = MODEL_MAP[model_name](in_dim=graph["x"].shape[1], **ctor_kwargs)
        result = train_graph(model, graph, model_params, train_cfg, device=device)
        val_keys = {"auc", "f1", "f1_threshold", "accuracy", "rmse", "mae", "r2"}
        val_result = {}
        std_result = {}
        for k, v in result.items():
            if k.startswith("std_test_"):
                std_key = k[len("std_test_"):]
                std_result[std_key] = v
            elif k in val_keys:
                val_result[k] = v
        if val_result:
            print(f"val: {val_result}")
        if std_result:
            print(f"std_test: {std_result}")
        return


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

    cat_cols = [c for c in X_train.columns if X_train[c].dtype.kind in ("O", "b")]

    print("-" * 60)
    print(f"Training {model_name}...")
    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Positive ratio: {y_train.mean():.4f}")
    print("-" * 60)

    if model_name in SEQUENCE_MODELS:
        from dppbench.models import LSTMForecaster, GRUForecaster, TransformerForecaster
        MODEL_MAP = {
            "LSTM": LSTMForecaster,
            "GRU": GRUForecaster,
            "Transformer": TransformerForecaster,
        }
        # Sequence models expect a pure numeric matrix; coerce remaining object cols.
        X_train_num = X_train.copy()
        X_val_num = X_val.copy()
        for c in cat_cols:
            X_train_num[c] = pd.to_numeric(X_train_num[c], errors="coerce")
            X_val_num[c] = pd.to_numeric(X_val_num[c], errors="coerce")
        X_train_num = X_train_num.fillna(0.0)
        X_val_num = X_val_num.fillna(0.0)
        model_params = {**model_params, "device": device}
        model = MODEL_MAP[model_name](**model_params)
        model.fit(X_train_num, y_train, X_val=X_val_num, y_val=y_val)
        metrics = train_cfg.get("metrics", ["rmse"])
        val_result = model.evaluate(X_val_num, y_val, metrics=metrics)
    elif model_name in TABULAR_DEEP_MODELS:
        MODEL_MAP = {
            "MLP": MLP,
            "TabTransformer": TabTransformer,
            "FTTransformer": FTTransformer,
            "SAINT": SAINT,
        }
        model_params = {**model_params, "device": device}
        model = MODEL_MAP[model_name](**model_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        metrics = train_cfg.get(
            "metrics", ["auc"] if model_params.get("task", "binary") == "binary" else ["rmse"],
        )
        val_result = model.evaluate(X_val, y_val, metrics=metrics)
    else:
        if model_name != "LightGBM":
            raise ValueError(f"Unsupported tabular model '{model_name}'.")
        model = LightGBMModel(**model_params)
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            sample_weight=train_sample_weight,
            eval_sample_weight=[val_sample_weight] if val_sample_weight is not None else None,
            categorical_features=cat_cols or "auto",
        )
        metrics = train_cfg.get("metrics", ["auc"])
        val_result = model.evaluate(X_val, y_val, metrics=metrics)
    print(f"val: {val_result}")

    std_test_df = splits.get("std_test")
    if std_test_df is not None and len(std_test_df) > 0 and target_col in std_test_df.columns:
        X_std = std_test_df.drop(columns=drop_cols, errors="ignore")
        for c in X_train.columns:
            if c not in X_std.columns:
                X_std[c] = 0
        X_std = X_std[X_train.columns]
        y_std = std_test_df[target_col].values
        if model_name in SEQUENCE_MODELS:
            X_std_num = X_std.copy()
            for c in cat_cols:
                if c in X_std_num.columns:
                    X_std_num[c] = pd.to_numeric(X_std_num[c], errors="coerce")
            X_std_num = X_std_num.fillna(0.0)
            std_result = model.evaluate(X_std_num, y_std, metrics=metrics)
        else:
            std_result = model.evaluate(X_std, y_std, metrics=metrics)
        print(f"std_test: {std_result}")


def main():
    args = parse_args()
    device = _resolve_device(args.gpu_id)
    task_dir = os.path.join(BASE_DIR, args.data_name)
    pre_process_yaml = os.path.join(task_dir, "pre_process.yaml")
    model_yaml = os.path.join(task_dir, "model.yaml")
    cfg = _resolve_model_config(_load_yaml(model_yaml), args.model)

    model_name = cfg.get("model")
    paths = _output_paths(args.output_dir, args.data_name, model_name)
    old_stdout = sys.stdout
    log_file = None
    error = None
    if paths:
        log_file = open(paths["log"], "w", encoding="utf-8")
        sys.stdout = _Tee(old_stdout, log_file)
        print(f"[output] writing run artifacts to {paths['run_dir']}")
    try:
        _ensure_std_test(args, cfg)
        if model_name in {"LightGBM"} | SEQUENCE_MODELS | TABULAR_DEEP_MODELS | GRAPH_MODELS or args.data_name in TABULAR_REGISTRY:
            _train_tabular(args, cfg, pre_process_yaml, device=device)
        elif model_name in REC_MODELS or args.data_name in REC_DATASETS:
            _train_rec(args, cfg, pre_process_yaml, device=device)
        else:
            raise ValueError(
                f"Cannot route training for data_name='{args.data_name}', "
                f"model='{model_name}'."
            )
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if paths:
            sys.stdout.flush()
            sys.stdout = old_stdout
            log_file.close()
            _write_run_artifacts(paths, args, cfg, model_name, device, error=error)


if __name__ == "__main__":
    main()
