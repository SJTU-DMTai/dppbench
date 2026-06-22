"""Command-line entry for the AlphaClean baseline.

Examples::

    # Tabular smoke test
    python -m baselines.AlphaClean.run_alphaclean \\
        --data_name fraud_detection --n_iters 3 --beam_width 3 \\
        --batch_per_iter 4 --small_n 1500

    # Recommendation smoke test
    python -m baselines.AlphaClean.run_alphaclean \\
        --data_name movielens --n_iters 3 --beam_width 3 \\
        --batch_per_iter 4 --small_n 1500
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.AlphaClean.alphaclean import AlphaClean  # noqa: E402

BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    p = argparse.ArgumentParser(description="AlphaClean baseline runner")
    p.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    p.add_argument("--data_name", type=str, default="movielens", choices=SUPPORTED)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--beam_width", type=int, default=None)
    p.add_argument("--n_iters", type=int, default=None)
    p.add_argument("--batch_per_iter", type=int, default=None)
    p.add_argument("--gamma", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--small_n", type=int, default=None,
                   help="Subsample size during search. 0 disables.")
    p.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                   default=None,
                   help="Skip the final full-data evaluation.")
    p.add_argument("--no_pruner", dest="learned_pruning", action="store_false",
                   default=None,
                   help="Disable the learned-pruning component.")
    p.add_argument("--pruner_min_samples", type=int, default=None)
    p.add_argument("--pruner_refit_every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no_fast_train", dest="fast_train", action="store_false",
                   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    runner = AlphaClean(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        beam_width=args.beam_width,
        n_iters=args.n_iters,
        batch_per_iter=args.batch_per_iter,
        gamma=args.gamma,
        patience=args.patience,
        small_n=args.small_n,
        eval_full=args.eval_full,
        learned_pruning=args.learned_pruning,
        pruner_min_samples=args.pruner_min_samples,
        pruner_refit_every=args.pruner_refit_every,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        fast_train=args.fast_train,
        config_path=args.config,
    )
    result = runner.run()

    print("\n" + "=" * 60)
    print("AlphaClean Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Best fitness:     {fit_str}")
    print(f"Best metrics:     {result.get('best_metrics')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Pruner:           {result.get('pruner_path')}")
    print(f"Unique evals:     {result.get('n_unique_evaluations')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("\nSearch history:")
    for rec in result.get("search_history", []):
        it = rec["iter"]
        bf = rec["best_fitness"]
        mf = rec["mean_fitness"]
        nc = rec["n_candidates"]
        np_ = rec["n_pruned"]
        ops = rec["best_ops"]
        print(f"  iter={it:>3}  best={bf:.4f}  mean={mf:.4f}  "
              f"cands={nc} pruned={np_}  ops={ops}")
    print("=" * 60)


if __name__ == "__main__":
    main()
