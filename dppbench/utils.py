import os
import re
import importlib
import yaml
import pandas as pd
from .models import FNN, DeepFM, DIN, DIEN, SIM, build_feature_columns


MODEL_MAP = {
    "FNN": FNN,
    "DeepFM": DeepFM,
    "DIN": DIN,
    "DIEN": DIEN,
    "SIM": SIM,
}

DATASET_MAP = {
    "amazon_beauty": "dppbench.tasks.amazon_beauty.amazon_beauty_data.AmazonBeautyData",
    "kuairec": "dppbench.tasks.kuairec.kuairec_data.KuairecData",
    "movielens": "dppbench.tasks.movielens.movielens_data.MovielensData",
    "tenrec": "dppbench.tasks.tenrec.tenrec_data.TenrecData",
    "yelp": "dppbench.tasks.yelp.yelp_data.YelpData",
}


def load_model_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_class(data_name):
    if data_name not in DATASET_MAP:
        raise ValueError(
            f"Unknown rec dataset '{data_name}'. Available: {sorted(DATASET_MAP)}"
        )
    module_path, class_name = DATASET_MAP[data_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_data(data_name, data_dir, pipeline_yaml, cfg):
    dataset_cls = _load_dataset_class(data_name)
    data = dataset_cls(data_dir=data_dir)
    if hasattr(data, "set_model_config"):
        data.set_model_config(cfg)
    data.load_data()
    data.run_pre_process(os.path.abspath(pipeline_yaml))
    splits = data.split()
    all_df = pd.concat(list(splits.values()), ignore_index=True)
    feature_columns = build_feature_columns(all_df, cfg, col_types=data.col_types)
    return [splits, feature_columns]


def get_model(feature_columns, cfg):
    model_name = cfg["model"]
    model_cls = MODEL_MAP[model_name]

    model_params = cfg.get("model_params", {})
    model = model_cls(
        dnn_feature_columns=feature_columns,
        dnn_hidden_units=tuple(model_params.get("dnn_hidden_units", (256, 128))),
        att_hidden_size=tuple(model_params.get("att_hidden_size", (64, 16))),
        gru_hidden_size=model_params.get("gru_hidden_size"),
        sim_top_k=model_params.get("sim_top_k", 10),
        dnn_dropout=model_params.get("dnn_dropout", 0.0),
        l2_reg_embedding=model_params.get("l2_reg_embedding", 1e-6),
        seed=model_params.get("seed", 1024),
        task=model_params.get("task", "binary"),
        device=model_params.get("device", "cpu"),
    )
    return model
