"""Command line entry point for the SPIO baseline.

Examples:

    # Tabular smoke test (uses project apikeys.json)
    python -m baselines.SPIO.run_spio \
        --data_name fraud_detection --llm_model deepseek-v4-flash \
        --n_candidates 2 --small_n 1000 --downstream_eval_n 500 --no_eval_full

    # Recommendation smoke test
    python -m baselines.SPIO.run_spio \
        --data_name movielens --llm_model deepseek-v4-flash \
        --n_candidates 2 --small_n 2000 --downstream_eval_n 1000 --no_eval_full
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.SPIO.spio import SPIO  # noqa: E402


BASE_DIR = os.path.join(_ROOT, "dppbench", "tasks")
SUPPORTED = ["home_credit", "fraud_detection", "amazon_beauty",
             "movielens", "yelp", "tenrec"]


def parse_args():
    parser = argparse.ArgumentParser(description="SPIO baseline runner")
    parser.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    parser.add_argument("--data_name", type=str, default="movielens",
                        choices=SUPPORTED)
    parser.add_argument("--data_dir", type=str, default=None)

    # ---- LLM ----
    parser.add_argument("--llm_backend", type=str, default=None,
                        choices=["api", "local"],
                        help="api = OpenAI-compatible HTTP endpoint; "
                             "local = HF transformers (lazy loaded).")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="Model name (api) or local path / repo (local).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Disabled; credentials are loaded from apikeys.json.")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Disabled; endpoint is loaded from apikeys.json.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    # ---- SPIO loop ----
    parser.add_argument("--n_candidates", type=int, default=None,
                        help="Number of CustomOp snippets generated per "
                             "stage; the highest-AUC candidate is kept "
                             "(SPIO-S, default 2).")
    parser.add_argument("--max_retry", type=int, default=None,
                        help="Same-call retry budget when the LLM reply "
                             "fails to parse (default 2).")
    parser.add_argument("--stage_max_per_cat", type=int, default=None)

    # ---- Eval ----
    parser.add_argument("--small_n", type=int, default=None,
                        help="Subsample size used by the SANDBOX during "
                             "the agent loop (0 disables; the final "
                             "evaluation always uses the full dataset).")
    parser.add_argument("--no_eval_full", dest="eval_full", action="store_false",
                        default=None,
                        help="Skip the final downstream-model evaluation.")
    parser.add_argument("--downstream_eval_n", type=int, default=None,
                        help="Subsample size used by the per-candidate "
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

    runner = SPIO(
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
        n_candidates=args.n_candidates,
        max_retry=args.max_retry,
        stage_max_per_cat=args.stage_max_per_cat,
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
    print("SPIO Final Report")
    print("=" * 60)
    print(f"Dataset:          {args.data_name}")
    print(f"Task type:        {result.get('task_type')}")
    fit = result.get("best_fitness")
    fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
    print(f"Downstream fit:   {fit_str}")
    print(f"Downstream met:   {result.get('best_metrics')}")
    in_loop_fit = result.get("best_fitness_in_loop")
    in_loop_str = f"{in_loop_fit:.4f}" if isinstance(in_loop_fit, float) else "n/a"
    print(f"In-loop best:     {in_loop_str}")
    print("Per-stage chosen fitness:")
    for stage, score in (result.get("per_stage_chosen_fitness") or {}).items():
        score_str = f"{score:.4f}" if isinstance(score, float) else "n/a"
        print(f"    {stage:<22s}{score_str}")
    if result.get("eval_error"):
        print(f"Eval error:       {result.get('eval_error')}")
    print(f"Final ops:        {result.get('final_pipeline_ops')}")
    print(f"Best pipeline:    {result.get('best_pipeline_path')}")
    print(f"NL plan:          {result.get('nl_plan_path')}")
    print(f"Trajectory:       {result.get('trajectory_path')}")
    print(f"Run summary:      {result.get('run_summary_path')}")
    print(f"Duration:         {result.get('duration_seconds'):.1f}s")
    print(f"Output dir:       {result.get('output_dir')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
