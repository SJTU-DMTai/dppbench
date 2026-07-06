"""Canonical task data loading helpers shared by baseline infrastructure."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Callable, Optional

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


def import_class(dotted_path):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class TaskDataLoader:
    """Load and cache the raw DPPBench task data object for a baseline run."""

    def __init__(
        self,
        data_name: str,
        data_dir: Optional[str] = None,
        model_config_provider: Optional[Callable[[], dict]] = None,
    ):
        if data_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset: {data_name}. Available: {list(DATASET_REGISTRY.keys())}"
            )
        self.data_name = data_name
        self.data_dir = data_dir
        self.model_config_provider = model_config_provider
        self.task_type, self.data_cls_path = DATASET_REGISTRY[data_name]
        self._data = None

    def resolved_data_dir(self):
        if not self.data_dir:
            return None
        data_dir = os.path.abspath(self.data_dir)
        task_data_dir = os.path.join(data_dir, self.data_name, "data")
        if os.path.isdir(task_data_dir):
            return task_data_dir
        return data_dir

    def make_data_instance(self):
        cls = import_class(self.data_cls_path)
        return cls(data_dir=self.resolved_data_dir())

    def configure_data_instance(self, data):
        if self.model_config_provider and hasattr(data, "set_model_config"):
            data.set_model_config(self.model_config_provider())
        return data

    def make_configured_data_instance(self):
        return self.configure_data_instance(self.make_data_instance())

    def load_data(self):
        if self._data is None:
            self._data = self.make_configured_data_instance()
            self._data.load_data()
        return self._data

    def clear_cache(self):
        self._data = None
