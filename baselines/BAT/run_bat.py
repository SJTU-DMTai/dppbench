"""Command line entry point for the BAT baseline.

Examples:

    # Tabular smoke test (downstream feedback ON, uses project apikeys.json)
    python -m baselines.BAT.run_bat \
        --data_name fraud_detection --llm_model gpt-4o-mini \
        --max_rollout_steps 4 --max_depth 4 --small_n 3000

    # Tabular smoke test (downstream feedback OFF, recovers BAT-original)
    python -m baselines.BAT.run_bat \
        --data_name fraud_detection --llm_model gpt-4o-mini \
        --max_rollout_steps 4 --max_depth 4 --small_n 3000 \
        --no_use_downstream

    # Recommendation smoke test
    python -m baselines.BAT.run_bat \
        --data_name movielens --llm_model gpt-4o-mini \
        --max_rollout_steps 4 --max_depth 4 --small_n 3000
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.BAT.bat import BAT  # noqa: E402


BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="BAT baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens",
                        choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)

    # ---- LLM ----
    parser.add_argument("--llm_backend", type=str, default=None,
                        choices=["api", "local"])
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None,
                        help="Disabled; credentials are loaded from apikeys.json.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Disabled; endpoint is loaded from apikeys.json.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    # ---- MCTS ----
    parser.add_argument("--max_rollout_steps", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--max_chain_len", type=int, default=None)
    parser.add_argument("--c_puct", dest="exploration_constant",
                        type=float, default=None,
                        help="UCB1 exploration constant.")
    parser.add_argument("--early_stop_n_paths", type=int, default=None)
    parser.add_argument("--early_stop_eps", type=float, default=None)

    # ---- Reward fusion ----
    parser.add_argument("--reward_alpha", type=float, default=None,
                        help="weight for column_similarity (BAT-original).")
    parser.add_argument("--reward_beta", type=float, default=None,
                        help="weight for downstream-model fitness.")
    parser.add_argument("--reward_gamma", type=float, default=None,
                        help="weight for the LLM judge.")
    parser.add_argument("--no_use_downstream", dest="use_downstream",
                        action="store_false", default=None,
                        help="Disable the downstream-model reward channel "
                             "and recover BAT's original target-instance-free "
                             "behaviour.")
    parser.add_argument("--use_llm_judge", action="store_true", default=None,
                        help="Enable the LLM-judge reward channel.")
    parser.add_argument("--columns_match_threshold", type=float, default=None)

    # ---- Eval ----
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample size used inside the SANDBOX during "
                             "MCTS (0 disables; the final evaluation always "
                             "uses the full dataset).")
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final downstream-model evaluation.")
    parser.add_argument("--downstream_eval_n", type=int, default=None,
                        help="Subsample size used by the in-loop downstream "
                             "evaluator (0 = full data, expensive).")
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

    runner = BAT(
        task_dir=task_dir,
        data_name=args.data_name,
        data_dir=args.data_dir,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_rollout_steps=args.max_rollout_steps,
        max_depth=args.max_depth,
        max_chain_len=args.max_chain_len,
        exploration_constant=args.exploration_constant,
        early_stop_n_paths=args.early_stop_n_paths,
        early_stop_eps=args.early_stop_eps,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        use_downstream=args.use_downstream,
        use_llm_judge=args.use_llm_judge,
        columns_match_threshold=args.columns_match_threshold,
        small_n=args.small_n,
        eval_full=args.eval_full,
        downstream_eval_n=args.downstream_eval_n,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        fast_train=args.fast_train,
        config_path=args.config,
    )
    result = runner.run()

    print("\n" + "=" * 60)
    print("BAT Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    print(f"Task type:        {result.get('task_type')}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Downstream fit:   {fit_str}")
    print(f"Downstream met:   {result.get('best_metrics')}")
    if result.get("eval_error"):
        print(f"Eval error:       {result.get('eval_error')}")
    print(f"Rollouts:         {result.get('n_rollouts')}")
    print(f"Best paths:       {result.get('n_best_paths')}")
    print(f"Pipeline legal:   {result.get('is_legal')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Agent log:        {result.get('agent_log_path')}")
    print(f"Search tree:      {result.get('tree_path')}")
    print(f"Best paths:       {result.get('best_paths_path')}")
    duration = result.get("duration_seconds") or 0.0
    print(f"Duration:         {duration:.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
