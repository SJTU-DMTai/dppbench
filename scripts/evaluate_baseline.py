"""End-to-end baseline evaluation harness using the per-task standard
test set (std-test).

For each (task, baseline) pair this script does the following sequentially:

1. **Ensure std-test exists**: if ``dppbench/tasks/<task>/std_test/`` is
   missing it calls :mod:`scripts.build_std_test` to build it (idempotent;
   re-running with the same seed produces the same files).
2. **Run the baseline's pipeline construction loop** (default ``SAGA``)
   from scratch — the task's bundled ``pre_process.yaml`` is *not*
   reused; every baseline starts from an empty pipeline.
3. **Re-evaluate** the produced ``best_pipeline.yaml`` through SAGA's
   ``PipelineEvaluator`` so that downstream model training is run end-to-
   end with the std-test rows attached. The executor (see
   :mod:`baselines.common.executor`) automatically applies the baseline pipeline,
   trains the downstream model, and reports the model's metrics on the
   pre-frozen std-test set together with the inference wall-time.
4. **Aggregate**: prints a table and writes a CSV with one row per
   (task, baseline) including the std-test metric and inference time.

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
import os
import sys
import time
import traceback
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the std-test builder and the existing baseline registry so we
# don't duplicate task lists / baseline configs.
from scripts.build_std_test import (  # noqa: E402
    TASK_REGISTRY as STD_TEST_TASKS,
    run_for_task as build_std_test_for_task,
)


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
        "kwargs": {},
    },
    "SPIO": {
        "cls_path": "baselines.SPIO.spio.SPIO",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "ReAct": {
        "cls_path": "baselines.ReAct.react.ReAct",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "BAT": {
        "cls_path": "baselines.BAT.bat.BAT",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "DataMaster": {
        "cls_path": "baselines.DataMaster.data_master.DataMaster",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "AutoPrep": {
        "cls_path": "baselines.AutoPrep.auto_prep.AutoPrep",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "DeepPrep": {
        "cls_path": "baselines.DeepPrep.deepprep.DeepPrep",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "Learn2Clean": {
        "cls_path": "baselines.Learn2Clean.learn2clean.Learn2Clean",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "AlphaClean": {
        "cls_path": "baselines.AlphaClean.alphaclean.AlphaClean",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "DiffPrep": {
        "cls_path": "baselines.DiffPrep.diffprep.DiffPrep",
        "supports_eval_full": True,
        "kwargs": {},
    },
    "CtxPipe": {
        "cls_path": "baselines.CtxPipe.ctxpipe.CtxPipe",
        "supports_eval_full": True,
        "kwargs": {},
    },
}


def _std_test_dir(task_name: str) -> str:
    return os.path.join(_ROOT, "dppbench", "tasks", task_name, "std_test")


def _std_test_present(task_name: str) -> bool:
    d = _std_test_dir(task_name)
    if not os.path.isdir(d):
        return False
    # std_test.parquet is required for every task type.
    return os.path.isfile(os.path.join(d, "std_test.parquet"))


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
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_ROOT, "outputs", "eval_std_test"),
    )
    parser.add_argument(
        "--output_csv", type=str, default=None,
        help="Override CSV path (default: <output_dir>/results.csv).",
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


def _ensure_std_test(task_name: str, skip_build: bool, quiet: bool) -> dict:
    """Build std-test for the task if missing. Returns a small status dict."""
    if _std_test_present(task_name):
        return {"built": False, "ok": True}
    if skip_build:
        raise FileNotFoundError(
            f"std-test missing for '{task_name}' at {_std_test_dir(task_name)} "
            f"and --skip_build was passed; run scripts/build_std_test.py first."
        )
    if not quiet:
        print(f"[{task_name}] std-test not found; building now ...")
    build_std_test_for_task(task_name, dry_run=False)
    return {"built": True, "ok": _std_test_present(task_name)}


def _run_baseline_and_eval(
    task_name: str,
    baseline_name: str,
    args,
    out_root: str,
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
    if conf["supports_eval_full"]:
        kwargs["eval_full"] = False

    task_dir = os.path.join(_ROOT, "dppbench", "tasks", task_name)
    runner_output_dir = os.path.join(out_root, baseline_name, task_name)
    os.makedirs(runner_output_dir, exist_ok=True)

    runner = cls(
        task_dir=task_dir,
        data_name=task_name,
        data_dir=args.data_dir,
        seed=args.seed,
        output_dir=runner_output_dir,
        verbose=not args.quiet,
        device=args.device,
        **kwargs,
    )

    # ---- Phase 1: construct preprocessing pipeline ---------------------
    t0 = time.time()
    result = runner.run()
    construct_time = time.time() - t0

    pipeline_yaml = result.get("best_pipeline_yaml")
    if not pipeline_yaml:
        raise RuntimeError(
            f"{baseline_name} did not produce best_pipeline_yaml for {task_name}"
        )

    # ---- Phase 2: end-to-end downstream training + std-test scoring ----
    # Use SAGA's evaluator which delegates to baselines.common.executor.TrainingExecutor.
    # The executor already applies the pipeline, trains the downstream
    # model, and reports both val and std_test_* metrics (including
    # std_test_inference_seconds).
    from baselines.SAGA.evaluator import PipelineEvaluator
    from baselines.SAGA.pipeline import Pipeline

    pipeline = Pipeline.from_yaml(pipeline_yaml)
    evaluator = PipelineEvaluator(
        task_dir=task_dir, data_name=task_name,
        data_dir=args.data_dir, verbose=not args.quiet,
        device=args.device,
    )
    eval_t0 = time.time()
    ev = evaluator.evaluate(pipeline)
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

    return {
        "task": task_name,
        "baseline": baseline_name,
        "val_fitness": float(ev.fitness) if ev.success else None,
        "std_test_auc": float(std_test_auc) if std_test_auc is not None else None,
        "std_test_metric_name": std_test_primary_name,
        "std_test_metric": (
            float(std_test_primary) if std_test_primary is not None else None
        ),
        "std_test_inference_seconds": (
            float(std_test_metrics["inference_seconds"])
            if "inference_seconds" in std_test_metrics else None
        ),
        "std_test_n_rows": (
            int(std_test_metrics["n_rows"])
            if "n_rows" in std_test_metrics else None
        ),
        "construct_time_s": round(construct_time, 1),
        "eval_time_s": round(eval_time, 1),
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


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = [
        "task", "baseline", "val_fitness",
        "std_test_metric_name", "std_test_metric",
        "std_test_auc", "std_test_inference_seconds", "std_test_n_rows",
        "construct_time_s", "eval_time_s", "n_steps", "error",
    ]

    def fmt(row, key):
        v = row.get(key)
        if v is None:
            return "n/a"
        if isinstance(v, float):
            if key in ("val_fitness", "std_test_auc", "std_test_metric"):
                return f"{v:.4f}"
            return f"{v:.3f}"
        if isinstance(v, list):
            return "[" + ",".join(str(x) for x in v) + "]"
        return str(v)

    cells = [[fmt(r, h) for h in headers] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(head)
    print(sep)
    for c in cells:
        print("| " + " | ".join(c[i].ljust(widths[i]) for i in range(len(headers))) + " |")


def _write_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "task", "baseline", "val_fitness",
        "std_test_metric_name", "std_test_metric",
        "std_test_auc", "std_test_inference_seconds", "std_test_n_rows",
        "construct_time_s", "eval_time_s", "n_steps", "ops",
        "n_unique_evals", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if isinstance(out.get("ops"), list):
                out["ops"] = "|".join(out["ops"])
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
    csv_path = args.output_csv or os.path.join(out_root, "results.csv")

    print("=" * 60)
    print(f"Std-test baseline evaluation")
    print(f"  baseline:  {args.baseline}")
    print(f"  tasks:     {tasks}")
    print(f"  output:    {out_root}")
    print("=" * 60)

    rows: list[dict[str, Any]] = []
    for task_name in tasks:
        print("\n" + "-" * 60)
        print(f"[{task_name}] baseline = {args.baseline}")
        print("-" * 60)

        row: dict[str, Any] = {"task": task_name, "baseline": args.baseline}
        try:
            _ensure_std_test(task_name, args.skip_build, args.quiet)
            row.update(_run_baseline_and_eval(
                task_name, args.baseline, args, out_root
            ))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            if not args.quiet:
                print(tb)
            row.update({
                "val_fitness": None,
                "std_test_metric_name": None,
                "std_test_metric": None,
                "std_test_auc": None,
                "std_test_inference_seconds": None,
                "std_test_n_rows": None,
                "construct_time_s": None,
                "eval_time_s": None,
                "n_steps": None,
                "ops": None,
                "n_unique_evals": None,
                "error": f"{type(e).__name__}: {e}",
            })
        rows.append(row)
        print(
            f"[{task_name}] {args.baseline}: "
            f"std_test_metric={row.get('std_test_metric')} "
            f"({row.get('std_test_metric_name')})  "
            f"std_test_inference_s={row.get('std_test_inference_seconds')}  "
            f"construct={row.get('construct_time_s')}s  "
            f"eval={row.get('eval_time_s')}s  "
            f"err={row.get('error')}"
        )

    _write_csv(rows, csv_path)
    print("\n" + "=" * 60)
    print(f"Results CSV: {csv_path}")
    print("=" * 60)
    _print_table(rows)


if __name__ == "__main__":
    main()
