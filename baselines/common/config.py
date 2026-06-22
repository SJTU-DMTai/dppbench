"""Shared helpers for baseline-level hyperparameter configs."""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import yaml


CONFIG_FILE_NAME = "config.yaml"


def default_config_path(module_file: str) -> str:
    """Return the default config path next to a baseline module file."""
    return os.path.join(os.path.dirname(os.path.abspath(module_file)), CONFIG_FILE_NAME)


def load_baseline_config(config_path: str, required_keys: Iterable[str]) -> dict[str, Any]:
    """Load and validate one baseline config file."""
    required = tuple(required_keys)
    required_set = set(required)
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Baseline config must be a non-empty mapping: {path}")

    keys = set(data)
    missing = [key for key in required if key not in keys]
    unknown = sorted(keys - required_set)
    if missing:
        raise ValueError(
            f"Baseline config {path} is missing required keys: {missing}"
        )
    if unknown:
        raise ValueError(f"Baseline config {path} has unknown keys: {unknown}")
    return data


def resolve_config_value(config: dict[str, Any], key: str, override: Any) -> Any:
    """Use an explicit override when present, otherwise use config[key]."""
    return config[key] if override is None else override
