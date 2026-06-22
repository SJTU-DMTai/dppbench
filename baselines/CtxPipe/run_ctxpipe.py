"""Command line entry point for the CtxPipe baseline.

Examples:

    # Tabular smoke test
    python -m baselines.CtxPipe.run_ctxpipe \
        --data_name fraud_detection --n_episodes 3 --max_steps 5 --small_n 3000

    # Recommendation smoke test
    python -m baselines.CtxPipe.run_ctxpipe \
        --data_name movielens --n_episodes 3 --max_steps 5 --small_n 3000
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.CtxPipe.ctxpipe import CtxPipe  # noqa: E402


BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="CtxPipe baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens",
                        choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Number of RL episodes (smoke test: 3-5).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum operators per pipeline.")
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample size during RL training. "
                             "0 disables subsampling.")
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final full-data re-evaluation.")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--eps_start", type=float, default=None)
    parser.add_argument("--eps_end", type=float, default=None)
    parser.add_argument("--eps_decay_episodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--target_update_every", type=int, default=None)
    parser.add_argument("--min_buffer", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--buffer_capacity", type=int, default=None)
    parser.add_argument("--illegal_penalty", type=float, default=None)
    parser.add_argument("--failure_reward", type=float, default=None)
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

    runner = CtxPipe(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        small_n=args.small_n,
        eval_full=args.eval_full,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        batch_size=args.batch_size,
        target_update_every=args.target_update_every,
        min_buffer=args.min_buffer,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        buffer_capacity=args.buffer_capacity,
        illegal_penalty=args.illegal_penalty,
        failure_reward=args.failure_reward,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        fast_train=args.fast_train,
        config_path=args.config,
    )
    result = runner.run()

    print("\n" + "=" * 60)
    print("CtxPipe Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Best fitness:     {fit_str}")
    print(f"Best metrics:     {result.get('best_metrics')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Q-network:        {result.get('q_network_path')}")
    print(f"Unique evals:     {result.get('n_unique_evaluations')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("\nRL history:")
    for h in result.get("rl_history", []):
        ep = h["episode"]
        rwd = h["reward"]
        ok = h["success"]
        f_v = h["fitness"]
        f_str = f"{f_v:.4f}" if isinstance(f_v, float) else "n/a"
        ops = h["ops"]
        print(f"  ep={ep:>3}  reward={rwd:+.3f}  success={ok}  "
              f"fit={f_str}  ops={ops}")
    print("=" * 60)


if __name__ == "__main__":
    main()
