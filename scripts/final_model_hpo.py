"""Final-stage validation-only tuning; never called by pipeline search."""
import json
import inspect
import math
import time
from pathlib import Path

from baselines.common.evaluator import PipelineEvaluator
from baselines.common.executor import (
    GRAPH_MODELS,
    REC_MODELS,
    TABULAR_DEEP_MODELS,
)


def _model_params(configs):
    return [{"model_params": params} for params in configs]


def candidate_configs(model_name):
    """Return the fixed v2 candidates shared by all baselines for a model."""
    if model_name == "LightGBM":
        return _model_params([
            {"learning_rate": .03, "n_estimators": 1200, "num_leaves": 31,
             "max_depth": -1, "min_child_samples": 20, "subsample": 1., "subsample_freq": 0,
             "colsample_bytree": 1., "reg_alpha": 0., "reg_lambda": 0.},
            {"learning_rate": .05, "n_estimators": 700, "num_leaves": 31,
             "max_depth": -1, "min_child_samples": 20, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": .1},
            {"learning_rate": .1, "n_estimators": 400, "num_leaves": 31,
             "max_depth": -1, "min_child_samples": 20, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": 1.},
            {"learning_rate": .03, "n_estimators": 1200, "num_leaves": 63,
             "max_depth": -1, "min_child_samples": 20, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": .1},
            {"learning_rate": .05, "n_estimators": 700, "num_leaves": 63,
             "max_depth": 10, "min_child_samples": 50, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": 1.},
            {"learning_rate": .1, "n_estimators": 400, "num_leaves": 63,
             "max_depth": 10, "min_child_samples": 50, "subsample": 1., "subsample_freq": 0,
             "colsample_bytree": .8, "reg_alpha": 1., "reg_lambda": .1},
            {"learning_rate": .03, "n_estimators": 1200, "num_leaves": 127,
             "max_depth": 12, "min_child_samples": 50, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .6, "reg_alpha": 1., "reg_lambda": 1.},
            {"learning_rate": .05, "n_estimators": 700, "num_leaves": 127,
             "max_depth": 12, "min_child_samples": 100, "subsample": .8, "subsample_freq": 1,
             "colsample_bytree": .8, "reg_alpha": 1., "reg_lambda": 5.},
            {"learning_rate": .1, "n_estimators": 400, "num_leaves": 127,
             "max_depth": -1, "min_child_samples": 100, "subsample": 1., "subsample_freq": 0,
             "colsample_bytree": .6, "reg_alpha": 5., "reg_lambda": 5.},
        ])

    if model_name in REC_MODELS:
        tuples = [
            (.0003, 8, [128, 64], 0., 1e-6, 0., 256),
            (.0003, 16, [256, 128], .2, 1e-6, 1e-5, 512),
            (.0003, 32, [512, 256], .3, 1e-5, 1e-4, 1024),
            (.001, 8, [128, 64], .2, 1e-6, 1e-5, 512),
            (.001, 16, [256, 128], 0., 1e-6, 0., 256),
            (.001, 32, [512, 256], .3, 1e-5, 1e-4, 1024),
            (.003, 8, [128, 64], .3, 1e-5, 1e-4, 1024),
            (.003, 16, [256, 128], .2, 1e-6, 1e-5, 256),
            (.003, 32, [512, 256], 0., 1e-6, 0., 512),
        ]
        return [
            {
                "model_params": {
                    "embedding_dim": emb,
                    "dnn_hidden_units": hidden,
                    "dnn_dropout": dropout,
                    "l2_reg_embedding": l2_embedding,
                },
                "train": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                },
            }
            for lr, emb, hidden, dropout, l2_embedding, weight_decay, batch_size
            in tuples
        ]

    if model_name in GRAPH_MODELS:
        return _model_params([
            {"lr": lr, "hidden_dim": hidden, "num_layers": layers,
             "dropout": dropout, "weight_decay": weight_decay}
            for lr, hidden, layers, dropout, weight_decay in [
                (.001, 32, 1, 0., 0.), (.001, 64, 2, .3, 5e-4),
                (.001, 128, 3, .5, 1e-3), (.005, 32, 2, .3, 5e-4),
                (.005, 64, 3, .5, 1e-3), (.005, 128, 1, 0., 0.),
                (.01, 32, 3, .5, 1e-3), (.01, 64, 1, 0., 0.),
                (.01, 128, 2, .3, 5e-4),
            ]
        ])

    if model_name in {"LSTM", "GRU"}:
        return _model_params([
            {"lr": lr, "hidden_size": hidden, "num_layers": layers,
             "dropout": dropout, "batch_size": batch_size}
            for lr, hidden, layers, dropout, batch_size in [
                (.0003, 32, 1, 0., 128), (.0003, 64, 2, .2, 256),
                (.0003, 128, 2, .3, 512), (.001, 32, 2, .2, 256),
                (.001, 64, 1, 0., 128), (.001, 128, 2, .3, 512),
                (.003, 32, 2, .3, 512), (.003, 64, 2, .2, 128),
                (.003, 128, 1, 0., 256),
            ]
        ])

    if model_name == "Transformer":
        return _model_params([
            {"lr": lr, "d_model": dim, "nhead": heads, "num_layers": layers,
             "dim_feedforward": feedforward, "dropout": dropout,
             "batch_size": batch_size}
            for lr, dim, heads, layers, feedforward, dropout, batch_size in [
                (.0003, 32, 4, 1, 64, 0., 128),
                (.0003, 64, 4, 2, 128, .1, 256),
                (.0003, 128, 8, 3, 256, .2, 512),
                (.001, 32, 4, 2, 128, .1, 256),
                (.001, 64, 4, 1, 128, 0., 128),
                (.001, 128, 8, 3, 512, .2, 512),
                (.003, 32, 4, 3, 128, .2, 512),
                (.003, 64, 8, 2, 256, .1, 128),
                (.003, 128, 8, 1, 256, 0., 256),
            ]
        ])

    if model_name == "MLP":
        return _model_params([
            {"lr": lr, "hidden_units": hidden, "dropout": dropout,
             "weight_decay": weight_decay, "batch_size": batch_size}
            for lr, hidden, dropout, weight_decay, batch_size in [
                (.0003, [128, 64], 0., 0., 128),
                (.0003, [256, 128], .2, 1e-5, 256),
                (.0003, [512, 256], .3, 1e-4, 512),
                (.001, [128, 64], .2, 1e-5, 256),
                (.001, [256, 128], 0., 0., 128),
                (.001, [512, 256], .3, 1e-4, 512),
                (.003, [128, 64], .3, 1e-4, 512),
                (.003, [256, 128], .2, 1e-5, 128),
                (.003, [512, 256], 0., 0., 256),
            ]
        ])

    if model_name in TABULAR_DEEP_MODELS:
        return _model_params([
            {"lr": lr, "d_model": dim, "nhead": heads,
             "num_layers": layers, "dim_feedforward": feedforward,
             "dropout": dropout, "weight_decay": weight_decay,
             "batch_size": batch_size}
            for lr, dim, heads, layers, feedforward, dropout, weight_decay, batch_size in [
                (.0003, 32, 4, 1, 64, 0., 0., 128),
                (.0003, 64, 4, 2, 128, .1, 1e-5, 256),
                (.0003, 128, 8, 3, 256, .2, 1e-4, 512),
                (.001, 32, 4, 2, 128, .1, 1e-5, 256),
                (.001, 64, 4, 1, 128, 0., 0., 128),
                (.001, 128, 8, 3, 512, .2, 1e-4, 512),
                (.003, 32, 4, 3, 128, .2, 1e-4, 512),
                (.003, 64, 8, 2, 256, .1, 1e-5, 128),
                (.003, 128, 8, 1, 256, 0., 0., 256),
            ]
        ])

    raise ValueError(f"No final HPO space is defined for model {model_name!r}")


def resolve_evaluator_class(pipeline, evaluator_class=None):
    """Keep custom checkpoint formats on their matching evaluation path."""
    provider = getattr(pipeline, 'get_evaluator_class', None)
    required = provider() if callable(provider) else PipelineEvaluator
    if evaluator_class is None:
        return required
    if required is not PipelineEvaluator and not issubclass(evaluator_class, required):
        raise TypeError(
            f'{type(pipeline).__name__} requires {required.__module__}.{required.__name__}; '
            f'got {evaluator_class.__module__}.{evaluator_class.__name__}. '
            'Use the baseline-specific evaluator for both final HPO and evaluation.'
        )
    return evaluator_class


def tune_final_model(pipeline, *, output_dir, evaluator_class=None, **kwargs):
    evaluator_class = resolve_evaluator_class(pipeline, evaluator_class)
    started = time.monotonic()
    trials = []
    best = None
    best_score = -math.inf
    evaluator_name = f'{evaluator_class.__module__}.{evaluator_class.__name__}'
    evaluator_source = inspect.getfile(evaluator_class)
    summary = {"space_version": "common_hparams_v2", "trials": trials,
               "evaluator_class": evaluator_name, "evaluator_source": evaluator_source}
    if kwargs.get('verbose', False):
        print(f'[final HPO] evaluator={evaluator_name} source={evaluator_source}')
    path = Path(output_dir) / "final_hpo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    for params in candidate_configs(kwargs["model_name"]):
        cache_before = pipeline.performance_stats() if hasattr(pipeline, 'performance_stats') else {}
        evaluator = evaluator_class(**kwargs, model_config=params)
        try:
            evaluator.set_evaluate_std_test(False)
            cfg = evaluator._executor.get_model_config()
            metric = cfg["train"]["metrics"][0]
            result = evaluator.evaluate(pipeline, scoring_method="full_train")
            value = result.metrics.get(metric)
            valid = (
                result.success
                and not result.error
                and value is not None
                and math.isfinite(float(value))
            )
            cache_after = pipeline.performance_stats() if hasattr(pipeline, 'performance_stats') else {}
            cache_delta = {device: {key: value-cache_before.get(device, {}).get(key, 0)
                                   for key, value in stats.items()
                                   if key not in ('cached_bytes', 'peak_cached_bytes')}
                           for device, stats in cache_after.items()}
            trials.append({
                "params": params,
                "rec_protocol": result.metrics.get('rec_protocol'),
                "metric": metric,
                "value": value,
                "success": valid,
                "error": result.error,
                "seconds": result.duration_seconds,
                "transform_cache": cache_after,
                "transform_cache_delta": cache_delta,
            })
            if valid:
                minimize = metric in {"rmse", "mse", "mae", "logloss"}
                score = -float(value) if minimize else float(value)
                if score > best_score:
                    best_score, best = score, params
                    summary['best_rec_protocol'] = result.metrics.get('rec_protocol')
        finally:
            evaluator._executor.cleanup()
        summary.update(
            best_params=best,
            seconds=time.monotonic() - started,
        )
        path.write_text(json.dumps(summary, indent=2, default=str))
    if best is None:
        raise RuntimeError(f"All final HPO trials failed; see {path}")
    # Use a fresh model for final frozen-test evaluation. The standard test
    # was never evaluated by any candidate trial.
    final_evaluator = evaluator_class(**kwargs, model_config=best)
    protocol = summary.get('best_rec_protocol') or {}
    final_evaluator._executor.expected_rec_schema_sha256 = protocol.get('feature_schema_sha256')
    final_evaluator._executor.expected_category_maps_sha256 = protocol.get('category_maps_sha256')
    return final_evaluator, summary
