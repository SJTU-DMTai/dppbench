"""Command line entry point for the SAGA baseline.

Example:

    python -m baselines.SAGA.run_saga \
        --data_name movielens \
        --population_size 8 \
        --n_generations 3 \
        --top_k 3 \
        --n_physical_trials 3
"""
from __future__ import annotations

import argparse
import os
import sys

# Make repo root importable when invoked as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.SAGA.saga import SAGA  # noqa: E402


BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec", "bondora"]


def parse_args():
    parser = argparse.ArgumentParser(description="SAGA baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens", choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--n_generations", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--n_physical_trials", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--mutation_rate", type=float, default=None)
    parser.add_argument("--crossover_rate", type=float, default=None)
    parser.add_argument("--tournament_size", type=int, default=None)
    parser.add_argument("--elitism_size", type=int, default=None)
    parser.add_argument("--random_pipeline_p_optional", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample rows during search (0 = full data).")
    parser.add_argument("--no_fast_train", dest="fast_train", action="store_false",
                        default=None,
                        help="Disable search-time fast training overrides.")
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip full-data final evaluation after search; useful for smoke tests.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    saga = SAGA(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        population_size=args.population_size,
        n_generations=args.n_generations,
        top_k=args.top_k,
        n_physical_trials=args.n_physical_trials,
        early_stop_patience=args.early_stop_patience,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        tournament_size=args.tournament_size,
        elitism_size=args.elitism_size,
        random_pipeline_p_optional=args.random_pipeline_p_optional,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        small_n=args.small_n,
        fast_train=args.fast_train,
        config_path=args.config,
        skip_final_eval=args.skip_final_eval,
    )
    result = saga.run()

    print("\n" + "=" * 60)
    print("SAGA Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    print(f"Best fitness:     {result['best_fitness']:.4f}")
    print(f"Best metrics:     {result['best_metrics']}")
    print(f"Best pipeline:    {result['best_pipeline_path']}")
    print(f"Unique evals:     {result['n_unique_evaluations']}")
    print(f"Duration:         {result['duration_seconds']:.1f}s")
    print(f"Output dir:       {result['output_dir']}")
    print("\nTop-K:")
    for i, t in enumerate(result["top_k"]):
        print(f"  #{i+1}  fit={t['fitness']:.4f}  ops={t['ops']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
