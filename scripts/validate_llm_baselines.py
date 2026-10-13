#!/usr/bin/env python3
"""Validate the basic integration of every LLM-driven DPPBench baseline.

The default ``examples`` mode loads the configured language model or semantic
encoder and runs each baseline on small recommendation and tabular examples.
For BAT/DeepPrep it validates prompt -> response -> parse -> execute; for
CtxPipe/FlowPipe it validates language-model context encoding -> policy search
-> pipeline. Costly final-model/std-test training is omitted. ``preflight``
remains available for offline imports and protocol tests.

Examples
--------
Real LLM examples on an explicitly selected idle GPU::

    python scripts/validate_llm_baselines.py --gpu-id 0

Offline checks only::

    python scripts/validate_llm_baselines.py --mode preflight
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = Path("/mnt/sdb/dengjiale/dppbench/data")
OUTPUT_DIR = Path("/mnt/sdb/dengjiale/dppbench/output/llm_baseline_validation")

# Keep this list aligned with the LLM baseline group in scripts/exps/common.sh.
LLM_BASELINES: dict[str, str] = {
    "BAT": "baselines.BAT.bat.BAT",
    "DeepPrep": "baselines.DeepPrep.deepprep.DeepPrep",
    "CtxPipe": "baselines.CtxPipe.ctxpipe.CtxPipe",
    "FlowPipe": "baselines.FlowPipe.flowpipe.FlowPipe",
}

MODEL_CONFIG_KEYS = {
    "BAT": ("llm_backend", "llm_model"),
    "DeepPrep": ("llm_backend", "llm_model"),
    "CtxPipe": ("gte_model_path",),
    "FlowPipe": ("llama_model_path", "context_cache_dir", "checkpoint_path"),
}

PROTOCOL_TESTS = (
    "baselines.DeepPrep.test_deepprep_protocol",
    "baselines.BAT.test_bat_alignment",
)

REQUIRED_INIT_ARGS = {
    "task_dir",
    "data_name",
    "data_dir",
    "output_dir",
    "eval_full",
    "model_name",
}


def _load_object(path: str) -> Any:
    module_name, object_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def _result(name: str, check: str, ok: bool, detail: str = "") -> dict[str, Any]:
    marker = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker}] {name}: {check}{suffix}", flush=True)
    return {"baseline": name, "check": check, "ok": ok, "detail": detail}


def _selected_baselines(value: str) -> list[str]:
    if value.lower() == "all":
        return list(LLM_BASELINES)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(LLM_BASELINES))
    if unknown:
        raise SystemExit(
            f"Unknown LLM baseline(s): {unknown}. Available: {list(LLM_BASELINES)}"
        )
    return selected


def run_preflight(selected: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    evaluator = importlib.import_module("scripts.evaluate_baseline")

    for name in selected:
        try:
            cls = _load_object(LLM_BASELINES[name])
            results.append(_result(name, "import", True, LLM_BASELINES[name]))
        except Exception as exc:
            results.append(_result(name, "import", False, f"{type(exc).__name__}: {exc}"))
            continue

        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        missing = sorted(REQUIRED_INIT_ARGS - params)
        results.append(
            _result(
                name,
                "constructor contract",
                not missing,
                "all common arguments present" if not missing else f"missing={missing}",
            )
        )

        config_path = ROOT / "baselines" / name / "config.yaml"
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            problems = [
                f"{key} is absent from config"
                for key in MODEL_CONFIG_KEYS[name]
                if key not in config
            ]
            if "api_key" in config and config.get("api_key") is not None:
                problems.append("api_key must remain null (loaded from apikeys.json)")
            results.append(
                _result(
                    name,
                    "LLM config",
                    not problems,
                    "; ".join(problems) if problems else str(config_path.relative_to(ROOT)),
                )
            )
        except Exception as exc:
            results.append(
                _result(name, "LLM config", False, f"{type(exc).__name__}: {exc}")
            )

        registered = evaluator.BASELINE_CONFIGS.get(name, {}).get("cls_path")
        results.append(
            _result(
                name,
                "shared evaluator registration",
                registered == LLM_BASELINES[name],
                f"cls_path={registered!r}",
            )
        )

    for module in PROTOCOL_TESTS:
        owner = module.split(".")[1]
        if owner not in selected:
            continue
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", module],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        results.append(
            _result(
                owner,
                "protocol unit tests",
                completed.returncode == 0,
                detail[-1] if detail else f"exit={completed.returncode}",
            )
        )
    return results


def _gpu_state(gpu_id: int) -> dict[str, int | str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if int(fields[0]) == gpu_id:
            return {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_used_mb": int(fields[2]),
                "memory_total_mb": int(fields[3]),
                "utilization_pct": int(fields[4]),
            }
    raise RuntimeError(f"GPU {gpu_id} was not reported by nvidia-smi")


def _worker_summary_path(output_dir: Path, baseline: str, task: str) -> Path:
    return output_dir / baseline / task / "validation_summary.json"


def run_example_worker(args: argparse.Namespace) -> int:
    """Run one real baseline/task example in an isolated process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from baselines.common.budget import Budget, budget_scope
    from baselines.common.pipeline import Pipeline, check_pipeline
    from baselines.common.utils import get_data, get_data_ctx
    from scripts import evaluate_baseline as evaluator
    from scripts.build_std_test import _load_model_config

    name, task = args.worker_baseline, args.worker_task
    summary_path = _worker_summary_path(args.output_dir, name, task)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"baseline": name, "task": task, "ok": False}
    try:
        conf = evaluator.BASELINE_CONFIGS[name]
        cls = _load_object(conf["cls_path"])
        kwargs = dict(conf["kwargs"])
        model_args = argparse.Namespace(
            ctxpipe_gte_model_path=str(args.ctxpipe_model_path),
            deepprep_model_path=str(args.deepprep_model_path),
            qwen3_model_path=str(args.qwen3_model_path),
        )
        evaluator._apply_baseline_model_paths(name, kwargs, model_args)
        if name == "FlowPipe" and args.flowpipe_checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(args.flowpipe_checkpoint_path)
        kwargs["eval_full"] = False
        if name == "BAT":
            kwargs["downstream_eval_n"] = args.bat_downstream_eval_n
        if name == "DeepPrep":
            kwargs["downstream_feedback"] = True
            kwargs["downstream_eval_n"] = args.deepprep_downstream_eval_n
        evaluator._apply_max_pipeline_len(conf, cls, kwargs, args.max_pipeline_len)

        smoke_overrides = {
            "small_n": args.small_n,
            "n_candidates": 1,
            "max_retry": 1,
            "max_turns": 2,
            "max_retry_per_turn": 1,
            "max_rollout_steps": 2,
            "n_action_samples": 1,
            "max_iterations": 2,
            "max_explore_turn": 2,
            "max_solution_attempts": 1,
            "frames": 2,
            "inference_samples": 1,
            "inference_top_n": 1,
        }
        init_params = inspect.signature(cls.__init__).parameters
        kwargs.update({k: v for k, v in smoke_overrides.items() if k in init_params})
        if name == "DeepPrep":
            kwargs["max_explore_turn"] = 6
        if name == "FlowPipe":
            kwargs.update(train_epochs=2, trajectories_per_epoch=2,
                          updates_per_epoch=1, trajectory_batch_size=2)
        if name == "CtxPipe" and args.ctxpipe_train_check:
            kwargs.update(frames=16, batch_size=2, logical_batch_size=1,
                          backpropagate_interval=1, fast_train=True)
        if name == "BAT":
            kwargs["max_rollout_steps"] = args.bat_max_rollouts
            kwargs["n_action_samples"] = args.bat_action_samples

        task_dir = ROOT / "dppbench" / "tasks" / task
        task_data_dir = Path(evaluator._task_data_dir(task, str(args.data_dir)))
        budget = Budget(
            max_pipeline_length=args.max_pipeline_len,
            max_evaluations=args.max_evaluations,
            max_llm_calls=args.max_llm_calls,
            max_seconds=args.max_seconds,
        )
        budget.start()
        with budget_scope(budget):
            runner = cls(
                task_dir=str(task_dir), data_name=task,
                data_dir=str(task_data_dir), seed=args.seed,
                output_dir=str(summary_path.parent / "run"), verbose=True,
                device="cuda:0",
                model_name=evaluator._task_model_name(task, None), **kwargs,
            )
            flow_context_path = None
            if name == "FlowPipe":
                flow_context_path = runner.precompute_context(
                    device="cuda:0",
                    max_length=args.context_max_length,
                    overwrite=True,
                )
            result = runner.run()

        yaml_text = result.get("best_pipeline_yaml")
        if not isinstance(yaml_text, str) or not yaml_text.strip():
            raise RuntimeError("baseline returned no best_pipeline_yaml")
        pipeline = Pipeline.from_yaml(yaml_text)
        roundtrip = pipeline.to_yaml()
        if roundtrip != Pipeline.from_yaml(roundtrip).to_yaml():
            raise RuntimeError("pipeline YAML is not stable after round-trip parsing")

        data = get_data(
            task,
            data_dir=str(task_data_dir),
            model_config_provider=lambda: _load_model_config(task),
        )
        report = check_pipeline(pipeline, get_data_ctx(task, data))
        snapshot = budget.snapshot()
        artifact_values = (
            result.get("agent_log_path"), result.get("trajectory_path"),
            result.get("run_summary_path"), result.get("q_network_path"),
            flow_context_path,
            Path(result.get("output_dir", "")) / "flowpipe_log.json",
        )
        artifacts = [
            str(path) for value in artifact_values
            if value and (path := Path(value)).is_file() and path.stat().st_size > 2
        ]
        errors = []
        if name in {"BAT", "DeepPrep"} and snapshot["llm_calls_used"] < 1:
            errors.append("no real LLM call was recorded")
        if not artifacts:
            errors.append("no non-empty model/trajectory artifact was produced")
        if not report.valid:
            errors.extend(report.errors)
        if name == "BAT" and runner.reward_cfg.use_downstream:
            metrics = result.get("best_metrics") or {}
            if result.get("best_fitness") is None:
                errors.append("BAT downstream task-metric evaluation failed: "
                              + str(metrics.get("downstream_error") or "no validation fitness"))
        if name == "DeepPrep" and runner.downstream_feedback:
            if result.get("best_fitness") is None:
                errors.append("DeepPrep sampled validation evaluation failed: no scored candidate")
        if name == "CtxPipe" and args.ctxpipe_train_check:
            stats = result.get('training_stats') or {}
            if not stats.get('component_optimizer_steps') or not stats.get('logical_optimizer_steps'):
                errors.append('CtxPipe did not update both component and logical policy optimizers')
            if not stats.get('component_parameter_changes') or not stats.get('logical_parameter_changes'):
                errors.append('CtxPipe policy parameter-change probes did not detect learning')
        if name == 'FlowPipe' and not (result.get('training_stats') or {}).get('policy_updates'):
            errors.append('FlowPipe task-specific policy performed no optimizer updates')
        summary.update({
            "ok": not errors, "errors": errors, "warnings": report.warnings,
            "llm_calls": snapshot["llm_calls_used"],
            "model_kind": "generative_llm" if name in {"BAT", "DeepPrep"} else "semantic_encoder",
            "evaluations": snapshot["evaluations_used"],
            "pipeline_ops": pipeline.op_names(),
            "pipeline_yaml": str(result.get("best_pipeline_path") or ""),
            "llm_artifacts": artifacts,
            "duration_seconds": result.get("duration_seconds"),
            "best_metrics": result.get("best_metrics") or {},
            "training_stats": result.get("training_stats") or {},
        })
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["ok"] else 1


def run_smoke(args: argparse.Namespace, selected: list[str]) -> list[dict[str, Any]]:
    if args.gpu_id is None:
        raise SystemExit("--gpu-id is required for examples mode; CPU fallback is not automatic")
    state = _gpu_state(args.gpu_id)
    print(f"Selected GPU: {json.dumps(state, ensure_ascii=False)}", flush=True)
    if not args.allow_busy_gpu and (
        int(state["memory_used_mb"]) > args.max_used_gpu_memory_mb
        or int(state["utilization_pct"]) > args.max_gpu_utilization
    ):
        raise SystemExit(
            "Selected GPU is busy. Choose an idle GPU or explicitly pass "
            "--allow-busy-gpu after checking other users/jobs."
        )

    tasks = [item.strip() for item in args.examples.split(",") if item.strip()]
    if not tasks:
        raise SystemExit("--examples must contain at least one task name")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name in selected:
        for task in tasks:
            summary_path = _worker_summary_path(args.output_dir, name, task)
            if summary_path.exists():
                summary_path.unlink()
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-baseline", name, "--worker-task", task,
                "--data-dir", str(args.data_dir),
                "--output-dir", str(args.output_dir),
                "--gpu-id", str(args.gpu_id),
                "--ctxpipe-model-path", str(args.ctxpipe_model_path),
                "--deepprep-model-path", str(args.deepprep_model_path),
                "--qwen3-model-path", str(args.qwen3_model_path),
                "--context-max-length", str(args.context_max_length),
                "--small-n", str(args.small_n),
                "--bat-downstream-eval-n", str(args.bat_downstream_eval_n),
                "--deepprep-downstream-eval-n", str(args.deepprep_downstream_eval_n),
                "--bat-max-rollouts", str(args.bat_max_rollouts),
                "--bat-action-samples", str(args.bat_action_samples),
                "--max-pipeline-len", str(args.max_pipeline_len),
                "--max-evaluations", str(args.max_evaluations),
                "--max-llm-calls", str(args.max_llm_calls),
                "--max-seconds", str(args.max_seconds),
                "--seed", str(args.seed),
            ]
            if args.ctxpipe_train_check:
                command.append('--ctxpipe-train-check')
            if args.flowpipe_checkpoint_path is not None:
                command.extend([
                    "--flowpipe-checkpoint-path",
                    str(args.flowpipe_checkpoint_path),
                ])
            print(f"\n[RUN] {name}/{task}\n  {' '.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=ROOT)
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                detail = (
                    f"llm_calls={summary.get('llm_calls')}, "
                    f"ops={summary.get('pipeline_ops')}, "
                    f"artifacts={summary.get('llm_artifacts')}"
                )
                if not summary.get("ok"):
                    detail = str(summary.get("errors") or summary.get("error") or detail)
                ok = bool(summary.get("ok")) and completed.returncode == 0
            except Exception as exc:
                ok = False
                detail = f"worker exit={completed.returncode}; summary error={exc}"
            results.append(_result(f"{name}/{task}", "real LLM example", ok, detail))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "examples", "all"), default="examples")
    parser.add_argument("--baselines", default="all", help="Comma-separated names or 'all'.")
    parser.add_argument(
        "--examples", "--task", dest="examples",
        default="amazon_beauty,fraud_detection",
        help="Comma-separated example tasks (default covers rec and tabular).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--max-used-gpu-memory-mb", type=int, default=1024)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--max-pipeline-len", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=48)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--small-n", type=int, default=128)
    parser.add_argument('--ctxpipe-train-check', action='store_true',
                        help='Run 16 policy steps with small batches and require real optimizer updates.')
    parser.add_argument("--deepprep-downstream-eval-n", type=int, default=10000,
                        help="DeepPrep sampled downstream-training size, separate from LLM observation size.")
    parser.add_argument("--bat-downstream-eval-n", type=int, default=10000,
                        help="BAT downstream-training sample size; separate from LLM observation --small-n.")
    parser.add_argument("--bat-max-rollouts", type=int, default=6)
    parser.add_argument("--bat-action-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ctxpipe-model-path", "--ctxpipe-gte-model-path",
        dest="ctxpipe_model_path", type=Path,
        default=Path("/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B"),
    )
    parser.add_argument(
        "--deepprep-model-path", type=Path,
        default=Path("/mnt/sdb/dengjiale/dppbench/llm_ckpt/DeepPrep-Qwen3-8B"),
    )
    parser.add_argument(
        "--qwen3-model-path", type=Path,
        default=Path("/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B"),
    )
    parser.add_argument("--context-max-length", type=int, default=1024)
    parser.add_argument("--flowpipe-checkpoint-path", type=Path, default=None)
    parser.add_argument("--worker-baseline", choices=tuple(LLM_BASELINES), help=argparse.SUPPRESS)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_baseline:
        if not args.worker_task or args.gpu_id is None:
            raise SystemExit("worker mode requires --worker-task and --gpu-id")
        return run_example_worker(args)
    selected = _selected_baselines(args.baselines)
    results: list[dict[str, Any]] = []
    if args.mode in {"preflight", "all"}:
        results.extend(run_preflight(selected))
    if args.mode in {"examples", "all"}:
        results.extend(run_smoke(args, selected))

    failed = [item for item in results if not item["ok"]]
    print(
        f"\nSummary: {len(results) - len(failed)}/{len(results)} checks passed; "
        f"{len(failed)} failed.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
