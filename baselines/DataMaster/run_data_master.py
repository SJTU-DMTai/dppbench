"""Command line entry point for the DataMaster baseline.

Examples:

    # Tabular smoke test (uses project apikeys.json)
    python -m baselines.DataMaster.run_data_master \
        --data_name fraud_detection --llm_model gpt-4o-mini \
        --max_iterations 3 --k_black 2 --max_chain_len 4 \
        --small_n 1000 --downstream_eval_n 500 --no_eval_full

    # Recommendation smoke test
    python -m baselines.DataMaster.run_data_master \
        --data_name movielens --llm_model gpt-4o-mini \
        --max_iterations 3 --k_black 2 --max_chain_len 4 \
        --small_n 1000 --downstream_eval_n 500 --no_eval_full

    # Local model
    python -m baselines.DataMaster.run_data_master \
        --data_name fraud_detection \
        --llm_backend local --llm_model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.DataMaster.data_master import DataMaster  # noqa: E402


BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="DataMaster baseline runner")
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

    # ---- Tree search ----
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Outer UCB-select loop iterations.")
    parser.add_argument("--k_black", type=int, default=None,
                        help="Number of black children spawned per parent "
                             "in each iteration.")
    parser.add_argument("--max_chain_len", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--max_err_cnt", type=int, default=None)
    parser.add_argument("--max_solution_attempts", type=int, default=None,
                        help="Per-black-child LLM retries on parse/legality "
                             "failure.")

    # ---- UCB ----
    parser.add_argument("--c_initial", type=float, default=None)
    parser.add_argument("--c_lower_bound", type=float, default=None)
    parser.add_argument("--decay", type=str, default=None,
                        choices=["linear", "exponential", "piecewise", "none"])
    parser.add_argument("--decay_alpha", type=float, default=None)
    parser.add_argument("--decay_gamma", type=float, default=None)
    parser.add_argument("--piecewise_t1", type=int, default=None)
    parser.add_argument("--piecewise_t2", type=int, default=None)
    parser.add_argument("--reward_kind", type=str, default=None,
                        choices=["fitness", "improvement"])

    # ---- Memory ----
    parser.add_argument("--memory_top_k", type=int, default=None)
    parser.add_argument("--memory_max_chars", type=int, default=None)

    # ---- Eval ----
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample size for the SANDBOX during agent "
                             "exploration (0 disables; final eval is always "
                             "full).")
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final downstream-model evaluation.")
    parser.add_argument("--downstream_eval_n", type=int, default=None,
                        help="Subsample size used by the agent-loop "
                             "downstream evaluator (0 = full data).")
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

    runner = DataMaster(
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
        max_iterations=args.max_iterations,
        k_black=args.k_black,
        max_chain_len=args.max_chain_len,
        max_depth=args.max_depth,
        max_err_cnt=args.max_err_cnt,
        max_solution_attempts=args.max_solution_attempts,
        c_initial=args.c_initial,
        c_lower_bound=args.c_lower_bound,
        decay=args.decay,
        decay_alpha=args.decay_alpha,
        decay_gamma=args.decay_gamma,
        piecewise_t1=args.piecewise_t1,
        piecewise_t2=args.piecewise_t2,
        reward_kind=args.reward_kind,
        memory_top_k=args.memory_top_k,
        memory_max_chars=args.memory_max_chars,
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
    print("DataMaster Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    print(f"Task type:        {result.get('task_type')}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Downstream fit:   {fit_str}")
    print(f"Downstream met:   {result.get('best_metrics')}")
    if result.get("eval_error"):
        print(f"Eval error:       {result.get('eval_error')}")
    print(f"Agent success:    {result.get('agent_success')}")
    print(f"Iterations:       {result.get('agent_n_iterations')}")
    print(f"Expansions:       {result.get('agent_n_expansions')}")
    print(f"Errors:           {result.get('agent_n_errors')}")
    print(f"Best node id:     {result.get('agent_best_node_id')}")
    print(f"Pipeline legal:   {result.get('is_legal')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"Agent log:        {result.get('agent_log_path')}")
    print(f"Search tree:      {result.get('tree_path')}")
    print(f"Memory log:       {result.get('memory_path')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
