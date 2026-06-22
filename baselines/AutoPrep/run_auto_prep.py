"""Command line entry point for the Auto-Prep baseline.

Example:

    python -m baselines.AutoPrep.run_auto_prep \
        --data_name fraud_detection \
        --n_iters 2 --beam 2 --n_candidates 3 --small_n 1500
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.AutoPrep.auto_prep import AutoPrep  # noqa: E402

BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-Prep baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens", choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--small_n", type=int, default=None)
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final full-data re-evaluation.")
    parser.add_argument("--eta", type=float, default=None,
                        help="Learning rate of the multiplicative-weights update.")
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_fast_train", dest="fast_train", action="store_false",
                        default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    runner = AutoPrep(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        n_iters=args.n_iters,
        beam=args.beam,
        n_candidates=args.n_candidates,
        max_depth=args.max_depth,
        small_n=args.small_n,
        eval_full=args.eval_full,
        eta=args.eta,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        fast_train=args.fast_train,
        config_path=args.config,
    )
    log = runner.run()

    print("\n" + "=" * 60)
    print("Auto-Prep Final Report")
    print("=" * 60)
    print(f"Dataset:                 {args.data_name}")
    print(f"Catalog ops:             {log['n_catalog_ops']}")
    print(f"Best small-data fitness: {log['best_small_fitness']}")
    print(f"Best full-data fitness:  {log['best_full_fitness']}")
    print(f"Best ops:                {log['best_ops']}")
    print(f"Best join edges:         {log['best_join_edges']}")
    print(f"Duration:                {log['duration_seconds']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
