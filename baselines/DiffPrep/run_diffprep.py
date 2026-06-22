"""Command-line entry for the DiffPrep baseline.

Examples::

    # Tabular smoke test
    python -m baselines.DiffPrep.run_diffprep \\
        --data_name fraud_detection --n_epochs 3 --small_n 2000

    # Recommendation smoke test
    python -m baselines.DiffPrep.run_diffprep \\
        --data_name movielens --n_epochs 3 --small_n 2000
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.DiffPrep.diffprep import DiffPrep  # noqa: E402

BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="DiffPrep baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens",
                        choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Bilevel training epochs (smoke: 3-5).")
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample size during search. 0 disables.")
    parser.add_argument("--lr_w", type=float, default=None,
                        help="Inner-loop learning rate (surrogate weights).")
    parser.add_argument("--lr_alpha", type=float, default=None,
                        help="Outer-loop learning rate (architecture parameters).")
    parser.add_argument("--eps_finite_diff", type=float, default=None,
                        help="Epsilon for the second-order finite difference (unused in first-order mode).")
    parser.add_argument("--flex", action="store_true", default=None,
                        help="Enable DiffPrep-Flex (learn operator order with Sinkhorn alpha).")
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final full-data evaluation.")
    parser.add_argument("--max_features", type=int, default=None,
                        help="Maximum number of numeric columns fed to the surrogate.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--continuous_init_scale", type=float, default=None)
    parser.add_argument("--second_order", action="store_true", default=None)
    parser.add_argument("--sgd_momentum", type=float, default=None)
    parser.add_argument("--surrogate_hidden_dim", type=int, default=None)
    parser.add_argument("--rec_emb_dim", type=int, default=None)
    parser.add_argument("--gumbel_tau", type=float, default=None)
    parser.add_argument("--no_hard_sample", dest="hard_sample",
                        action="store_false", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    runner = DiffPrep(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        n_epochs=args.n_epochs,
        small_n=args.small_n,
        lr_w=args.lr_w,
        lr_alpha=args.lr_alpha,
        eps_finite_diff=args.eps_finite_diff,
        flex=args.flex,
        eval_full=args.eval_full,
        max_features=args.max_features,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        continuous_init_scale=args.continuous_init_scale,
        second_order=args.second_order,
        sgd_momentum=args.sgd_momentum,
        surrogate_hidden_dim=args.surrogate_hidden_dim,
        rec_emb_dim=args.rec_emb_dim,
        gumbel_tau=args.gumbel_tau,
        hard_sample=args.hard_sample,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        config_path=args.config,
    )
    result = runner.run()

    print("\n" + "=" * 60)
    print("DiffPrep Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Best fitness:     {fit_str}")
    print(f"Best metrics:     {result.get('best_metrics')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Weights:          {result.get('weights_path')}")
    print(f"Unique evals:     {result.get('n_unique_evaluations')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("\nSearch history:")
    for rec in result.get("search_history", []):
        ep = rec["epoch"]
        tl = rec["train_loss"]
        vl = rec["val_loss"]
        va = rec["val_acc"]
        ops = rec["argmax_pipeline"]
        print(f"  epoch={ep+1:>3}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
              f"val_acc={va:.4f}  ops={ops}")
    print("=" * 60)


if __name__ == "__main__":
    main()
