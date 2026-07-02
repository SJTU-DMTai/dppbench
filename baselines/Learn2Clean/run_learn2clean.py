"""Command-line entry for the Learn2Clean baseline.

Examples::

    # Tabular smoke test
    python -m baselines.Learn2Clean.run_learn2clean \\
        --data_name fraud_detection --n_episodes 3 --max_steps 5 --small_n 1500

    # Recommendation smoke test (skip full-data eval to keep it quick)
    python -m baselines.Learn2Clean.run_learn2clean \\
        --data_name movielens --n_episodes 3 --max_steps 5 --small_n 1500 \\
        --no_eval_full
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.Learn2Clean.learn2clean import Learn2Clean  # noqa: E402

BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "bondora", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


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


def parse_args():
    p = argparse.ArgumentParser(description="Learn2Clean baseline runner")
    p.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    p.add_argument("--data_name", type=str, default="movielens", choices=SUPPORTED)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--small_n", type=int, default=None,
                   help="Subsample size during training. 0 disables.")
    p.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                   default=None,
                   help="Skip the final full-data evaluation.")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature_init", type=float, default=None)
    p.add_argument("--temperature_final", type=float, default=None)
    p.add_argument("--reward_max", type=float, default=None)
    p.add_argument("--illegal_reward", type=float, default=None)
    p.add_argument("--improvement_eps", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=-1,
                   help="GPU index to use (-1 = CPU). Sets CUDA_VISIBLE_DEVICES.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no_fast_train", dest="fast_train", action="store_false",
                   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    task_dir = os.path.join(BASE_DIR, args.data_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    device = _resolve_device(args.gpu_id)

    runner = Learn2Clean(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        small_n=args.small_n,
        eval_full=args.eval_full,
        gamma=args.gamma,
        lr=args.lr,
        temperature_init=args.temperature_init,
        temperature_final=args.temperature_final,
        reward_max=args.reward_max,
        illegal_reward=args.illegal_reward,
        improvement_eps=args.improvement_eps,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        device=device,
        fast_train=args.fast_train,
        config_path=args.config,
    )
    result = runner.run()

    print("\n" + "=" * 60)
    print("Learn2Clean Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Best fitness:     {fit_str}")
    print(f"Best metrics:     {result.get('best_metrics')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Q table:          {result.get('q_table_path')}")
    print(f"Unique evals:     {result.get('n_unique_evaluations')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Device:           {device}")
    print(f"GPU id:           {args.gpu_id}")
    print(f"Output dir:       {result.get('output_dir')}")
    print("\nTraining history:")
    for rec in result.get("search_history", []):
        ep = rec["episode"]
        T = rec["temperature"]
        rew = rec["reward"]
        fit = rec["fitness"]
        fit_s = f"{fit:.4f}" if fit is not None else "  n/a"
        ops = rec["ops"]
        print(f"  ep={ep:>3}  T={T:.3f}  reward={rew:+.3f}  "
              f"fitness={fit_s}  ops={ops}")
    print("=" * 60)


if __name__ == "__main__":
    main()
