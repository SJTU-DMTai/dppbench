"""Command-line entry point for the Random baseline."""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.Random.random_baseline import RandomBaseline  # noqa: E402

BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")


def parse_args():
    parser = argparse.ArgumentParser(description="Random baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="bondora")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--random_pipeline_p_optional", type=float, default=None)
    parser.add_argument("--small_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--no_fast_train", dest="fast_train", action="store_false", default=None)
    parser.add_argument("--skip_final_eval", dest="eval_full", action="store_false", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")
    runner = RandomBaseline(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        random_pipeline_p_optional=args.random_pipeline_p_optional,
        small_n=args.small_n,
        seed=args.seed,
        verbose=not args.quiet,
        device=args.device,
        fast_train=args.fast_train,
        eval_full=args.eval_full,
        model_name=args.model_name,
        config_path=args.config,
    )
    result = runner.run()
    fit = result["best_fitness"]
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print("\n" + "=" * 60)
    print("Random Baseline Final Report")
    print("=" * 60)
    print(f"Dataset:       {args.data_name}")
    print(f"Best fitness:  {fit_str}")
    print(f"Best metrics:  {result['best_metrics']}")
    print(f"Best pipeline: {result['best_pipeline_path']}")
    print(f"Unique evals:  {result['n_unique_evaluations']}")
    print(f"Duration:      {result['duration_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
