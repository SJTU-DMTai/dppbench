"""Shared training and checkpoint utilities for the three pipeline baselines."""
from __future__ import annotations

import os
from pathlib import Path

MODES = ("scratch", "pretrained", "finetune")


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"training_mode must be one of {MODES}, got {mode!r}")
    return mode


def pretrained_path(baseline, model, explicit=None, root=None, filename="checkpoint.pt"):
    root = root or os.environ.get("DPPBENCH_PRETRAINED_DIR", "checkpoints/pretrained")
    return str(Path(explicit or Path(root) / baseline / model / filename).expanduser().resolve())


def resolve_policy_source(baseline, model, explicit=None, root=None, training_mode='finetune'):
    filename = 'aipipe.pt' if baseline == 'HAIPipe' else 'checkpoint.pt'
    primary = pretrained_path(baseline, model, explicit, root, filename)
    if training_mode == 'scratch' or explicit or model not in ('GraphSAGE', 'GCN', 'GAT'):
        return model, primary, None
    files = (filename, 'alternative_evaluator.pt') if baseline == 'HAIPipe' else (filename,)
    for source_model in dict.fromkeys((model, 'LightGBM', 'DIN')):
        path = pretrained_path(baseline, source_model, root=root, filename=filename)
        if all((Path(path).parent / name).is_file() and (Path(path).parent / name).stat().st_size > 0
               for name in files):
            reason = None if source_model == model else f'{model} policy bundle missing; using {source_model} policy'
            return source_model, path, reason
    raise FileNotFoundError(f'Pretrained agent missing: {primary}; no complete native, LightGBM or DIN policy bundle for {baseline}')


def contract(baseline, env, model, encoder=None):
    """IDs are part of the contract, not merely tensor dimensions."""
    from .op_spec import CATALOG
    result = {
        "version": 1, "baseline": baseline, "model": model,
        "operators": list(CATALOG), "reward": "bounded_score_or_reciprocal_error_v1",
        "encoder": encoder,
        "search_semantics": "primary_schema_random_subset_rec_optional_fe_v2",
    }
    if baseline == "CtxPipe":
        result.update(actions={k: list(v.candidates) for k, v in env.components.items()},
                      logical_orders=[list(v) for v in env.logical_pipelines],
                      state_version="ctx33_global_mask_gate_parameter_v1")
    elif baseline == "HAIPipe":
        result.update(actions=env.component_actions, logical_orders=env.logical_pipelines,
                      vocabulary=env.primitive_vocabulary, state_version="hai19_global_mask_v1")
    else:
        result.update(actions=[(a.gid, a.name) for a in env.actions],
                      state_dim=env.state_dim, action_dim=env.action_dim,
                      state_version="flow33_global_mask_v1")
    return result


def checkpoint_contract_differences(actual, expected, prefix=""):
    """Compare contracts structurally, retaining actionable field-level evidence."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(set(actual) | set(expected)):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual or key not in expected:
                differences.append(dict(field=field, saved=actual.get(key), expected=expected.get(key),
                                        missing_in="saved" if key not in actual else "expected"))
            else:
                differences.extend(checkpoint_contract_differences(actual[key], expected[key], field))
        return differences
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return [dict(field=prefix, saved=actual, expected=expected)]
        return [diff for i, (left, right) in enumerate(zip(actual, expected))
                for diff in checkpoint_contract_differences(left, right, f"{prefix}[{i}]")]
    return [] if actual == expected else [dict(field=prefix or "checkpoint_metadata", saved=actual, expected=expected)]


def load_pretrained(agent, path, expected, report_path=None):
    import torch
    if not Path(path).is_file():
        raise FileNotFoundError(f"Pretrained agent missing: {path}. Run scripts/pretrain_baselines.py first, "
                                "or explicitly choose --training_mode scratch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved = payload.get("checkpoint_metadata")
    differences = checkpoint_contract_differences(saved, expected)
    if differences:
        import json
        report = dict(checkpoint=str(path), compatible=False, differences=differences,
                      saved=saved, expected=expected)
        if report_path:
            destination = Path(report_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        details = "; ".join(f"{d['field']}: saved={str(d['saved'])[:160]!r}, "
                            f"expected={str(d['expected'])[:160]!r}" for d in differences[:12])
        hint = ("Encoder path differs: verify checkpoint identity before relocating metadata. "
                if any(d['field'] == 'encoder.path' for d in differences) else "")
        raise ValueError(f"Incompatible agent checkpoint: {path}; {details}. " + hint +
                         "Changed operator IDs/actions/state/reward/search semantics require matching "
                         "pretraining; do not bypass this check." +
                         (f" Full report: {report_path}" if report_path else ""))
    agent.load(path)
    agent.checkpoint_metadata = expected
    return payload.get("provenance", {})


def freeze_agent(agent):
    models = ([agent.model] if hasattr(agent, "model") else
              [getattr(m, "network", m) for m in agent.models.values()] +
              [getattr(agent.logical_model, "network", agent.logical_model)])
    for model in models:
        model.eval()
        model.requires_grad_(False)


def resolved_model(evaluator):
    return evaluator._executor.get_model_config()["model"]


def isolate_training_data(evaluator):
    """Expose only the frozen training pool to agents and their reward executor.

    Disabling test *scoring* alone does not remove held-out rows from raw profiles.
    Patch before the first load, below SearchPipelineEvaluator's sampling wrapper.
    The final benchmark evaluator is a separate instance and is not patched.
    """
    if not hasattr(evaluator, "_orig_make_data_instance"):
        return evaluator  # Lightweight evaluator fixtures without a data factory.
    if getattr(evaluator, "_training_data_isolated", False):
        return evaluator
    import pandas as pd
    original_factory = evaluator._orig_make_data_instance

    def make_isolated():
        data = original_factory()
        original_load = data.load_data
        data._skip_std_test_loading = True
        data.load_std_test_frozen = lambda: False

        def load(*args, **kwargs):
            original_load(*args, **kwargs)
            rec = evaluator.task_type == "rec"
            filename = "interaction_frozen.parquet" if rec else "train_frozen.parquet"
            directory = data._std_test_dir(data.data_dir)
            path = Path(directory or "") / filename
            if not directory or not path.is_file():
                raise FileNotFoundError(f"Policy training requires {path}; run scripts/build_std_test.py first")
            frame = pd.read_parquet(path)
            if not rec and hasattr(data, "_normalize_frozen_frame"):
                frame = data._normalize_frozen_frame(frame)
            if "__split__" in frame and not frame["__split__"].eq("train").all():
                raise ValueError(f"Frozen agent training pool contains held-out split markers: {path}")
            if rec:
                data.interaction_df = frame
            else:
                data.train_df = frame
                data.test_df = None
            data._validation_prepared = False
            prepare = getattr(data, "_prepare_validation", None)
            if callable(prepare):
                prepare()
            return (data.interaction_df if rec else data.train_df), getattr(data, "test_df", None)

        data.load_data = load
        return data

    evaluator._orig_make_data_instance = make_isolated
    evaluator._executor._make_data_instance = (evaluator._patched_make_data_instance
                                             if evaluator.small_n > 0 else make_isolated)
    evaluator._executor.clear_data_cache()
    evaluator._training_data_isolated = True
    return evaluator
