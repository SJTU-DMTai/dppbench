"""Positive search rewards; reported fitness remains in evaluator units."""
from __future__ import annotations

import math

ERROR_METRICS = frozenset({'rmse', 'mse', 'mae', 'logloss'})


def fitness_to_reward(fitness: float, metric_key: str | None) -> float:
    """Scale=1 reciprocal for errors; preserve bounded scores such as AUC."""
    value = float(fitness)
    if not math.isfinite(value):
        raise ValueError('Search reward requires finite fitness')
    metric = metric_key.lower() if isinstance(metric_key, str) else ''
    if metric in ERROR_METRICS:
        if value > 0:
            raise ValueError('Error-metric fitness must be negated by the evaluator')
        return 1.0 / (1.0 - value)
    if metric in {'auc', 'accuracy', 'acc', 'f1', 'precision', 'recall', 'ap', 'average_precision'}:
        return max(0.0, min(1.0, value))
    return 0.5 + math.atan(value) / math.pi
