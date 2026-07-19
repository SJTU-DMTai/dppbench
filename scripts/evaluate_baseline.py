"""End-to-end baseline evaluation harness using the per-task standard
test set (std-test).

For each (task, baseline) pair this script does the following sequentially:

1. **Ensure std-test exists**: if ``dppbench/tasks/<task>/std_test/`` is
   missing it calls :mod:`scripts.build_std_test` to build it (idempotent;
   re-running with the same seed produces the same files).
2. **Run the baseline's pipeline construction loop** (default ``SAGA``)
   from scratch — the task's bundled ``prepare.yaml`` is *not*
   reused; every baseline starts from an empty pipeline.
3. **Re-evaluate** the produced ``best_pipeline.yaml`` through SAGA's
   ``PipelineEvaluator`` so that downstream model training is run end-to-
   end with the std-test rows attached. The executor (see
   :mod:`baselines.common.executor`) automatically applies the baseline pipeline,
   trains the downstream model, and reports the model's metrics on the
   pre-frozen std-test set together with the inference wall-time.
4. **Aggregate**: prints JSON summaries and writes a CSV with one row per
   (task, baseline) including the std-test metric and timing breakdown.

Examples::

    # Default: SAGA on every task in build_std_test.TASK_REGISTRY
    python scripts/evaluate_with_std_test.py

    # Run a subset of tasks
    python scripts/evaluate_with_std_test.py \\
        --data_names fraud_detection,movielens

    # Override the baseline (must be a key of BASELINE_CONFIGS defined
    # below in this script)
    python scripts/evaluate_with_std_test.py --baseline CtxPipe
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import sys
import time
import traceback
from typing import Any

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the std-test builder and the existing baseline registry so we
# don't duplicate task lists / baseline configs.
from scripts.build_std_test import (  # noqa: E402
    REC_SPLIT_METHOD,
    TASK_REGISTRY as STD_TEST_TASKS,
    rec_cold_start_user_filter,
    run_for_task as build_std_test_for_task,
)
from baselines.common.budget import Budget, budget_scope  # noqa: E402


def _load_class(cls_path: str):
    """Import ``module.path:ClassName`` (or ``module.path.ClassName``) and
    return the class object.
    """
    import importlib
    if ":" in cls_path:
        module_path, class_name = cls_path.split(":", 1)
    else:
        module_path, class_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# Per-baseline runner configuration. Each entry tells the harness how to
# instantiate the baseline's top-level class, plus whether the constructor
# accepts an ``eval_full`` flag (used to switch off costly downstream
# training during pipeline construction; std-test scoring is performed
# separately in Phase 2).
BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "SAGA": {
        "cls_path": "baselines.SAGA.saga.SAGA",
        "supports_eval_full": False,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "SPIO": {
        "cls_path": "baselines.SPIO.spio.SPIO",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "ReAct": {
        "cls_path": "baselines.ReAct.react.ReAct",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "BAT": {
        "cls_path": "baselines.BAT.bat.BAT",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_chain_len",
        "kwargs": {},
    },
    "DataMaster": {
        "cls_path": "baselines.DataMaster.data_master.DataMaster",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_chain_len",
        "kwargs": {},
    },
    "AutoPrep": {
        "cls_path": "baselines.AutoPrep.auto_prep.AutoPrep",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_depth",
        "kwargs": {},
    },
    "DeepPrep": {
        "cls_path": "baselines.DeepPrep.deepprep.DeepPrep",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_chain_len",
        "kwargs": {},
    },
    "Learn2Clean": {
        "cls_path": "baselines.Learn2Clean.learn2clean.Learn2Clean",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_steps",
        "kwargs": {},
    },
    "AlphaClean": {
        "cls_path": "baselines.AlphaClean.alphaclean.AlphaClean",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "Random": {
        "cls_path": "baselines.Random.random_baseline.RandomBaseline",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "DiffPrep": {
        "cls_path": "baselines.DiffPrep.diffprep.DiffPrep",
        "supports_eval_full": True,
        "time_cost_kind": "training",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "CtxPipe": {
        "cls_path": "baselines.CtxPipe.ctxpipe.CtxPipe",
        "supports_eval_full": True,
        "time_cost_kind": "training",
        "max_len_arg": "max_steps",
        "kwargs": {},
    },
    "HAIPipe": {
        "cls_path": "baselines.HAIPipe.haipipe.HAIPipe",
        "supports_eval_full": True,
        "time_cost_kind": "training",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
    "FlowPipe": {
        "cls_path": "baselines.FlowPipe.flowpipe.FlowPipe",
        "supports_eval_full": True,
        "time_cost_kind": "search_inference",
        "max_len_arg": "max_pipeline_len",
        "kwargs": {},
    },
}


def _task_data_dir(task_name: str, data_dir: str | None = None) -> str | None:
    if not data_dir:
        return None
    return os.path.join(os.path.abspath(data_dir), task_name, "data")


def _std_test_dir(task_name: str, data_dir: str | None = None) -> str:
    if data_dir:
        return os.path.join(os.path.abspath(data_dir), task_name, "std_test")
    return os.path.join(_ROOT, "dppbench", "tasks", task_name, "std_test")


def _std_test_present(task_name: str, data_dir: str | None = None) -> bool:
    d = _std_test_dir(task_name, data_dir=data_dir)
    if not os.path.isdir(d):
        return False
    # std_test.parquet is required for every task type.
    return os.path.isfile(os.path.join(d, "std_test.parquet"))


def _load_std_test_meta(task_name: str, data_dir: str | None = None) -> dict:
    path = os.path.join(_std_test_dir(task_name, data_dir=data_dir), "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def _task_model_name(task_name: str, model_name: str | None = None) -> str:
    if model_name:
        return model_name
    path = os.path.join(_ROOT, "dppbench", "tasks", task_name, "model.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("model", "model")


def _path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return token or "unknown"


def _task_model_output_dir(
    output_root: str,
    baseline_name: str,
    task_name: str,
    model_name: str,
) -> str:
    return os.path.join(
        os.path.abspath(output_root),
        _path_token(baseline_name),
        f"{_path_token(task_name)}_{_path_token(model_name)}",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a baseline (default: SAGA) on the per-task standard "
            "test set. For each task: build std-test → run baseline → "
            "evaluate downstream model on std-test."
        )
    )
    parser.add_argument(
        "--data_names", type=str, default="amazon_beauty",
        help=(
            "Comma-separated task names. Defaults to all tasks listed in "
            "build_std_test.TASK_REGISTRY."
        ),
    )
    parser.add_argument(
        "--baseline", type=str, default="SAGA",
        help=(
            "Baseline name. Must be a key of BASELINE_CONFIGS defined in "
            "this script (default: SAGA)."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Override downstream model name from model.yaml/model_options. "
            "When omitted, each task uses its default model."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "Optional dataset root. When set, task files are read from "
            "<data_dir>/<data_name>/data and std_test from "
            "<data_dir>/<data_name>/std_test."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_ROOT, "outputs", "eval_std_test"),
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Override CSV path (default: <output_dir>/<baseline>/results.csv).",
    )
    parser.add_argument(
        "--max_pipeline_len", type=int, default=None,
        help=(
            "Maximum number of operators allowed in the generated pipeline. "
            "The harness maps this to each baseline's native length knob "
            "and rejects final pipelines that still exceed it."
        ),
    )
    parser.add_argument("--max_evaluations", type=int, default=None)
    parser.add_argument("--max_memory_mb", type=float, default=None)
    parser.add_argument("--max_gpu_memory_mb", type=float, default=None)
    parser.add_argument("--max_llm_calls", type=int, default=None)
    parser.add_argument("--max_seconds", type=float, default=None)
    parser.add_argument("--max_output_rows", type=int, default=None)
    parser.add_argument("--max_output_columns", type=int, default=None)
    parser.add_argument(
        "--ctxpipe_gte_model_path", type=str,
        default="/mnt/sdb/dengjiale/dppbench/llm_ckpt/gte-large",
        help="Local GTE checkpoint used by CtxPipe.",
    )
    parser.add_argument(
        "--deepprep_model_path", type=str,
        default="/mnt/sdb/dengjiale/dppbench/llm_ckpt/DeepPrep-Qwen3-8B",
        help="Released local DeepPrep checkpoint.",
    )
    parser.add_argument(
        "--qwen3_model_path", type=str,
        default="/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B",
        help="Local Qwen3-8B checkpoint used by BAT and FlowPipe.",
    )
    parser.add_argument(
        "--skip_build", action="store_true",
        help=(
            "Don't auto-build std-test if missing — fail loudly instead. "
            "Useful for CI to ensure std-tests are pre-built."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--gpu_id", type=int, default=-1,
        help="GPU index to use (-1 = CPU). Sets CUDA_VISIBLE_DEVICES.",
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


def _ensure_std_test(
    task_name: str,
    data_dir: str | None,
    skip_build: bool,
    quiet: bool,
) -> dict:
    """Build std-test for the task if missing. Returns a small status dict."""
    needs_build = not _std_test_present(task_name, data_dir=data_dir)
    rebuild_reason = "std-test not found" if needs_build else None
    if not needs_build and STD_TEST_TASKS.get(task_name, (None, None))[1] == "rec":
        meta = _load_std_test_meta(task_name, data_dir=data_dir)
        needs_build = meta.get("split_method") != REC_SPLIT_METHOD
        if needs_build:
            rebuild_reason = "std-test protocol changed"
        elif meta.get("cold_start_user_filter") != rec_cold_start_user_filter(task_name):
            needs_build = True
            rebuild_reason = "std-test cold-start user filter changed"
    if not needs_build:
        return {"built": False, "ok": True}
    if skip_build:
        raise FileNotFoundError(
            f"std-test missing or stale for '{task_name}' at "
            f"{_std_test_dir(task_name, data_dir=data_dir)} "
            f"and --skip_build was passed; run scripts/build_std_test.py first."
        )
    if not quiet:
        print(f"[{task_name}] {rebuild_reason}; building now ...")
    build_std_test_for_task(task_name, dry_run=False, data_dir=data_dir)
    return {"built": True, "ok": _std_test_present(task_name, data_dir=data_dir)}


def _apply_max_pipeline_len(
    conf: dict[str, Any],
    cls,
    kwargs: dict[str, Any],
    max_pipeline_len: int | None,
) -> None:
    if max_pipeline_len is None:
        return
    if max_pipeline_len <= 0:
        raise ValueError("--max_pipeline_len must be a positive integer")
    arg_name = conf.get("max_len_arg")
    if not arg_name:
        raise ValueError("baseline config is missing max_len_arg")
    if arg_name not in inspect.signature(cls).parameters:
        raise TypeError(
            f"{cls.__name__} does not accept configured max length "
            f"argument '{arg_name}'"
        )
    kwargs[arg_name] = int(max_pipeline_len)


def _apply_baseline_model_paths(
    baseline_name: str,
    kwargs: dict[str, Any],
    args,
) -> None:
    """Map experiment checkpoint arguments to each baseline's public API."""
    if baseline_name == "CtxPipe":
        kwargs["gte_model_path"] = args.ctxpipe_gte_model_path
    elif baseline_name == "DeepPrep":
        kwargs.update({
            "llm_backend": "local",
            "llm_model": args.deepprep_model_path,
        })
    elif baseline_name == "BAT":
        kwargs.update({
            "llm_backend": "local",
            "llm_model": args.qwen3_model_path,
        })
    elif baseline_name == "FlowPipe":
        kwargs.update({
            "llama_model_path": args.qwen3_model_path,
            "llm_name": "Qwen3-8B",
        })


def _run_baseline_and_eval(
    task_name: str,
    baseline_name: str,
    args,
    out_root: str,
    budget: Budget | None = None,
) -> dict[str, Any]:
    """Run the baseline construction loop, then re-evaluate the produced
    pipeline end-to-end so the downstream model is trained and scored on
    the frozen std-test rows.
    """
    if baseline_name not in BASELINE_CONFIGS:
        raise SystemExit(
            f"Unknown baseline '{baseline_name}'. "
            f"Available: {sorted(BASELINE_CONFIGS)}"
        )
    conf = BASELINE_CONFIGS[baseline_name]
    cls = _load_class(conf["cls_path"])
    kwargs = dict(conf["kwargs"])
    _apply_baseline_model_paths(baseline_name, kwargs, args)
    if conf["supports_eval_full"]:
        kwargs["eval_full"] = False
    max_pipeline_len = (
        budget.max_pipeline_length if budget is not None
        else args.max_pipeline_len
    )
    _apply_max_pipeline_len(conf, cls, kwargs, max_pipeline_len)

    task_dir = os.path.join(_ROOT, "dppbench", "tasks", task_name)
    task_data_dir = _task_data_dir(task_name, args.data_dir)
    model_name = _task_model_name(task_name, args.model)
    runner_output_dir = _task_model_output_dir(
        out_root, baseline_name, task_name, model_name
    )
    os.makedirs(runner_output_dir, exist_ok=True)

    if "model_name" in inspect.signature(cls).parameters:
        kwargs["model_name"] = model_name

    # ---- Phase 1: construct preprocessing pipeline ---------------------
    t0 = time.time()
    if budget is not None:
        budget.start()
    with budget_scope(budget):
        if budget is not None:
            budget.check_resources()
        runner = cls(
            task_dir=task_dir,
            data_name=task_name,
            data_dir=task_data_dir,
            seed=args.seed,
            output_dir=runner_output_dir,
            verbose=not args.quiet,
            device=args.device,
            **kwargs,
        )
        result = runner.run()
        if budget is not None:
            budget.check_resources()
    search_budget_snapshot = budget.snapshot() if budget is not None else None
    construct_time = time.time() - t0
    if result.get("unsupported_task_type"):
        error = result.get("error") or result.get("eval_error") or (
            f"unsupported_task_type={result['unsupported_task_type']}"
        )
        raise RuntimeError(error)

    pipeline_yaml = result.get("best_pipeline_yaml")
    if not pipeline_yaml:
        raise RuntimeError(
            f"{baseline_name} did not produce best_pipeline_yaml for {task_name}"
        )

    # ---- Phase 2: end-to-end downstream training + std-test scoring ----
    # Search-time baseline fitness may use fast/proxy objectives. The final
    # benchmark number is computed here: full training/replay on frozen
    # std-test rows, with validation fitness kept only as diagnostic context.
    from baselines.common.evaluator import PipelineEvaluator
    from baselines.common.pipeline import Pipeline

    pipeline = Pipeline.from_yaml(pipeline_yaml)
    if max_pipeline_len is not None and len(pipeline) > max_pipeline_len:
        raise RuntimeError(
            f"{baseline_name} produced pipeline length {len(pipeline)} "
            f"> max_pipeline_len={max_pipeline_len}"
        )
    evaluator = PipelineEvaluator(
        task_dir=task_dir, data_name=task_name,
        data_dir=task_data_dir, verbose=not args.quiet,
        device=args.device,
        model_name=model_name,
        scoring_method="full_train",
    )
    eval_t0 = time.time()
    ev = evaluator.evaluate(pipeline, scoring_method="full_train")
    eval_time = time.time() - eval_t0

    metrics = ev.metrics or {}
    std_test_metrics = {
        k[len("std_test_"):]: v for k, v in metrics.items()
        if k.startswith("std_test_")
    }
    std_test_error = None
    if evaluator.task_type == "graph" and not std_test_metrics:
        std_test_error = (
            "graph std_test evaluation is not supported by common executor"
        )
    std_test_auc = std_test_metrics.get("auc")
    primary_keys = ("auc", "rmse", "mse", "logloss", "mae")
    std_test_primary_name = next(
        (k for k in primary_keys if k in std_test_metrics), None
    )
    std_test_primary = (
        std_test_metrics.get(std_test_primary_name)
        if std_test_primary_name else None
    )
    if std_test_primary is None:
        # Fall back to any non-bookkeeping metric.
        for k, v in std_test_metrics.items():
            if k not in ("inference_seconds", "n_rows"):
                std_test_primary = v
                std_test_primary_name = k
                break
    if std_test_error is not None:
        std_test_primary_name = "unsupported_graph_std_test"

    baseline_time_kind = conf.get("time_cost_kind", "search_inference")
    construct_seconds = round(construct_time, 1)
    final_eval_seconds = round(eval_time, 1)
    final_inference_seconds = (
        float(std_test_metrics["inference_seconds"])
        if "inference_seconds" in std_test_metrics else None
    )
    final_training_seconds = (
        round(max(eval_time - final_inference_seconds, 0.0), 1)
        if final_inference_seconds is not None else None
    )

    return {
        "task": task_name,
        "baseline": baseline_name,
        "model": model_name,
        "output_dir": runner_output_dir,
        "max_pipeline_len": max_pipeline_len,
        "budget": search_budget_snapshot,
        "search_fitness": (
            float(result["best_fitness"])
            if isinstance(result.get("best_fitness"), (int, float)) else None
        ),
        "final_val_fitness": float(ev.fitness) if ev.success else None,
        "val_fitness": float(ev.fitness) if ev.success else None,
        "std_test_auc": float(std_test_auc) if std_test_auc is not None else None,
        "std_test_metric_name": std_test_primary_name,
        "std_test_metric": (
            float(std_test_primary) if std_test_primary is not None else None
        ),
        "std_test_inference_seconds": final_inference_seconds,
        "baseline_time_kind": baseline_time_kind,
        "baseline_search_inference_seconds": (
            construct_seconds
            if baseline_time_kind == "search_inference" else None
        ),
        "baseline_training_seconds": (
            construct_seconds if baseline_time_kind == "training" else None
        ),
        "final_model_training_seconds": final_training_seconds,
        "final_model_inference_seconds": final_inference_seconds,
        "final_eval_seconds": final_eval_seconds,
        "std_test_n_rows": (
            int(std_test_metrics["n_rows"])
            if "n_rows" in std_test_metrics else None
        ),
        "construct_time_s": construct_seconds,
        "eval_time_s": final_eval_seconds,
        "n_steps": len(pipeline),
        "ops": pipeline.op_names(),
        "n_unique_evals": result.get("n_unique_evaluations"),
        "baseline_is_legal": result.get("is_legal"),
        "baseline_eval_error": result.get("eval_error"),
        "std_test_error": std_test_error,
        "error": (
            std_test_error
            if std_test_error is not None
            else None if ev.success else (ev.error or "evaluation failed")
        ),
    }


def _row_console_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": row.get("task"),
        "baseline": row.get("baseline"),
        "model": row.get("model"),
        "max_pipeline_len": row.get("max_pipeline_len"),
        "std_test": {
            "metric_name": row.get("std_test_metric_name"),
            "metric": row.get("std_test_metric"),
            "auc": row.get("std_test_auc"),
            "n_rows": row.get("std_test_n_rows"),
        },
        "time_cost_seconds": {
            "baseline_time_kind": row.get("baseline_time_kind"),
            "baseline_search_inference": row.get(
                "baseline_search_inference_seconds"
            ),
            "baseline_training": row.get("baseline_training_seconds"),
            "final_model_training": row.get("final_model_training_seconds"),
            "final_model_inference": row.get("final_model_inference_seconds"),
            "final_eval_total": row.get("final_eval_seconds"),
        },
        "pipeline": {
            "n_steps": row.get("n_steps"),
            "ops": row.get("ops"),
            "n_unique_evals": row.get("n_unique_evals"),
        },
        "budget": row.get("budget"),
        "error": row.get("error"),
    }


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _write_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "task", "baseline", "model", "output_dir", "max_pipeline_len",
        "search_fitness",
        "final_val_fitness", "val_fitness", "std_test_metric_name",
        "std_test_metric", "std_test_auc", "std_test_inference_seconds",
        "std_test_n_rows", "baseline_time_kind",
        "baseline_search_inference_seconds", "baseline_training_seconds",
        "final_model_training_seconds", "final_model_inference_seconds",
        "final_eval_seconds", "construct_time_s", "eval_time_s",
        "n_steps", "ops", "n_unique_evals", "baseline_is_legal",
        "baseline_eval_error", "std_test_error", "budget", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if isinstance(out.get("ops"), list):
                out["ops"] = "|".join(out["ops"])
            if isinstance(out.get("budget"), dict):
                out["budget"] = json.dumps(out["budget"], ensure_ascii=False)
            writer.writerow({k: out.get(k) for k in fields})


def main():
    args = parse_args()
    args.device = _resolve_device(args.gpu_id)

    if args.data_names:
        tasks = [s.strip() for s in args.data_names.split(",") if s.strip()]
    else:
        tasks = sorted(STD_TEST_TASKS)

    unknown = [t for t in tasks if t not in STD_TEST_TASKS]
    if unknown:
        raise SystemExit(
            f"Unknown tasks: {unknown}. "
            f"Available: {sorted(STD_TEST_TASKS)}"
        )

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)
    baseline_root = os.path.join(out_root, _path_token(args.baseline))
    os.makedirs(baseline_root, exist_ok=True)
    csv_path = args.output_csv or os.path.join(baseline_root, "results.csv")
    model_names = {task: _task_model_name(task, args.model) for task in tasks}

    print("=" * 60)
    print(f"Std-test baseline evaluation")
    print(f"  baseline:  {args.baseline}")
    print(f"  tasks:     {tasks}")
    print(f"  models:    {model_names}")
    print(f"  output:    {baseline_root}")
    print("=" * 60)

    rows: list[dict[str, Any]] = []
    for task_name in tasks:
        model_name = model_names[task_name]
        run_output_dir = _task_model_output_dir(
            out_root, args.baseline, task_name, model_name
        )
        print("\n" + "-" * 60)
        print(f"[{task_name}] baseline = {args.baseline}, model = {model_name}")
        print("-" * 60)

        row: dict[str, Any] = {
            "task": task_name,
            "baseline": args.baseline,
            "model": model_name,
            "output_dir": run_output_dir,
            "max_pipeline_len": args.max_pipeline_len,
        }
        budget = Budget(
            max_pipeline_length=args.max_pipeline_len,
            max_evaluations=args.max_evaluations,
            max_memory_mb=args.max_memory_mb,
            max_gpu_memory_mb=args.max_gpu_memory_mb,
            max_llm_calls=args.max_llm_calls,
            max_seconds=args.max_seconds,
            max_output_rows=args.max_output_rows,
            max_output_columns=args.max_output_columns,
        )
        try:
            _ensure_std_test(task_name, args.data_dir, args.skip_build, args.quiet)
            row.update(_run_baseline_and_eval(
                task_name, args.baseline, args, out_root, budget=budget
            ))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            if not args.quiet:
                print(tb)
            row.update({
                "val_fitness": None,
                "max_pipeline_len": args.max_pipeline_len,
                "std_test_metric_name": None,
                "std_test_metric": None,
                "std_test_auc": None,
                "std_test_inference_seconds": None,
                "std_test_n_rows": None,
                "baseline_time_kind": BASELINE_CONFIGS.get(
                    args.baseline, {}
                ).get("time_cost_kind"),
                "baseline_search_inference_seconds": None,
                "baseline_training_seconds": None,
                "final_model_training_seconds": None,
                "final_model_inference_seconds": None,
                "final_eval_seconds": None,
                "construct_time_s": None,
                "eval_time_s": None,
                "n_steps": None,
                "ops": None,
                "n_unique_evals": None,
                "budget": budget.snapshot(),
                "error": f"{type(e).__name__}: {e}",
            })
        rows.append(row)
        _print_json({"run_result": _row_console_summary(row)})

    _write_csv(rows, csv_path)
    print("\n" + "=" * 60)
    _print_json({
        "results_csv": csv_path,
        "results": [_row_console_summary(row) for row in rows],
    })
    print("=" * 60)


if __name__ == "__main__":
    main()
