"""Top-level DeepPrep orchestrator.

Compared to :class:`baselines.SAGA.saga.SAGA` and
:class:`baselines.CtxPipe.ctxpipe.CtxPipe`, DeepPrep replaces the search
strategy with an LLM-driven *tree-based agentic reasoning* loop. The
:class:`TreeAgent` observes sandbox structural feedback during exploration
(``<operator>``); when ``downstream_feedback`` is enabled, every
successful ``<solution>`` attempt also receives a small-data downstream
metric so the LLM can iteratively refine the pipeline. A final full-data
re-evaluation is still produced via :class:`DeepPrepEvaluator`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal, repair
from baselines.common.context import _infer_rec_context, _infer_tabular_context

from .evaluator import DeepPrepEvaluator
from .llm_client import LLMClient
from .sandbox import Sandbox
from .tree_agent import TreeAgent


CONFIG_KEYS = (
    "llm_backend",
    "llm_model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
    "timeout",
    "max_explore_turn",
    "max_chain_len",
    "max_depth",
    "max_err_cnt",
    "small_n",
    "eval_full",
    "downstream_feedback",
    "downstream_eval_n",
    "max_solution_attempts",
    "seed",
    "fast_train",
)


class DeepPrep:
    """LLM-Powered, tree-based agent for data-preparation pipelines."""

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        # ---- LLM ----
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        # ---- Tree search ----
        max_explore_turn: Optional[int] = None,
        max_chain_len: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_err_cnt: Optional[int] = None,
        # ---- Eval ----
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        downstream_feedback: Optional[bool] = None,
        downstream_eval_n: Optional[int] = None,
        max_solution_attempts: Optional[int] = None,
        seed: Optional[int] = None,
        # ---- IO ----
        output_dir: Optional[str] = None,
        verbose: bool = True,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        llm_backend = resolve_config_value(cfg, "llm_backend", llm_backend)
        llm_model = resolve_config_value(cfg, "llm_model", llm_model)
        api_key = resolve_config_value(cfg, "api_key", api_key)
        base_url = resolve_config_value(cfg, "base_url", base_url)
        temperature = resolve_config_value(cfg, "temperature", temperature)
        max_tokens = resolve_config_value(cfg, "max_tokens", max_tokens)
        timeout = resolve_config_value(cfg, "timeout", timeout)
        max_explore_turn = resolve_config_value(
            cfg, "max_explore_turn", max_explore_turn
        )
        max_chain_len = resolve_config_value(cfg, "max_chain_len", max_chain_len)
        max_depth = resolve_config_value(cfg, "max_depth", max_depth)
        max_err_cnt = resolve_config_value(cfg, "max_err_cnt", max_err_cnt)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        downstream_feedback = resolve_config_value(
            cfg, "downstream_feedback", downstream_feedback
        )
        downstream_eval_n = resolve_config_value(
            cfg, "downstream_eval_n", downstream_eval_n
        )
        max_solution_attempts = resolve_config_value(
            cfg, "max_solution_attempts", max_solution_attempts
        )
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir

        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)

        self.max_explore_turn = int(max_explore_turn)
        self.max_chain_len = int(max_chain_len)
        self.max_depth = int(max_depth)
        self.max_err_cnt = int(max_err_cnt)

        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.downstream_feedback = bool(downstream_feedback)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.max_solution_attempts = int(max_solution_attempts)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "DeepPrep", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_llm(self) -> LLMClient:
        return LLMClient(
            backend=self.llm_backend,
            model=self.llm_model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def _infer_context(self, sandbox: Sandbox) -> DataContext:
        if sandbox.data is None:
            sandbox.reset()
        if sandbox.task_type == "rec":
            return _infer_rec_context(self.data_name, {}, sandbox.data)
        return _infer_tabular_context(self.data_name, {}, sandbox.data)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()

        # Build the agent-side downstream evaluator FIRST so it survives the
        # sandbox try/finally below (it owns its own TrainingExecutor and is
        # decoupled from the sandbox's data instance).
        agent_evaluator: Optional[DeepPrepEvaluator] = None
        agent_eval_fn = None
        if self.downstream_feedback:
            agent_evaluator = DeepPrepEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                small_n=self.downstream_eval_n,
                seed=self.seed,
                verbose=False,
                device=self.device,
            )
            if self.fast_train:
                agent_evaluator.set_fast_mode(True)

            def _agent_eval_fn(pipeline: Pipeline):
                return agent_evaluator.evaluate_for_agent(pipeline)

            agent_eval_fn = _agent_eval_fn

        # 1. Sandbox (also drives schema-based context inference)
        sandbox = Sandbox(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.small_n,
            seed=self.seed,
        )
        try:
            ctx = self._infer_context(sandbox)

            if self.verbose:
                print("=" * 60)
                print(f"[DeepPrep] dataset={self.data_name}  task={ctx.task_type}")
                print(f"[DeepPrep] numeric={len(ctx.numeric_cols)}  "
                      f"categorical={len(ctx.categorical_cols)}  "
                      f"list={len(ctx.list_cols)}  text={len(ctx.text_cols)}")
                print(f"[DeepPrep] target={ctx.target_col}  time={ctx.time_col}  "
                      f"id={ctx.id_col}  aux={ctx.aux_dfs}")
                print(f"[DeepPrep] llm={self.llm_backend}/{self.llm_model}  "
                      f"max_turn={self.max_explore_turn}  "
                      f"max_chain={self.max_chain_len}  "
                      f"max_depth={self.max_depth}")
                print(f"[DeepPrep] small_n={self.small_n or 'OFF'}  "
                      f"eval_full={self.eval_full}  "
                      f"downstream_feedback={self.downstream_feedback}  "
                      f"downstream_eval_n={self.downstream_eval_n or 'OFF'}  "
                      f"max_solution_attempts={self.max_solution_attempts}")
                print("=" * 60)

            # 2. LLM + Agent
            llm = self._build_llm()
            agent = TreeAgent(
                llm=llm,
                sandbox=sandbox,
                ctx=ctx,
                max_explore_turn=self.max_explore_turn,
                max_chain_len=self.max_chain_len,
                max_depth=self.max_depth,
                max_err_cnt=self.max_err_cnt,
                verbose=self.verbose,
                seed=self.seed,
                downstream_evaluator=agent_eval_fn,
                max_solution_attempts=self.max_solution_attempts,
            )
            run = agent.run()
            pipeline: Pipeline = run.pipeline

            # 3. Final structural repair (mandatory ops, canonical order)
            repair(pipeline, ctx.task_type, ctx)
            legal = is_legal(pipeline, ctx.task_type)

            if self.verbose:
                print(f"[DeepPrep] agent done. success={run.success}  "
                      f"turns={run.n_turns}  errors={run.n_errors}  "
                      f"steps={[s.op for s in pipeline.steps]}  "
                      f"legal={legal}  "
                      f"solution_attempts={len(run.solution_attempts)}")
        finally:
            sandbox.cleanup()

        # 4. Final downstream evaluation (NOT fed back to the agent).
        eval_metrics: dict = {}
        eval_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        if self.eval_full:
            evaluator = DeepPrepEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                verbose=self.verbose,
                small_n=0,
                seed=self.seed,
                device=self.device,
            )
            try:
                ev = evaluator.evaluate(pipeline)
                eval_metrics = dict(ev.metrics or {})
                eval_fitness = ev.fitness if ev.success else None
                eval_error = ev.error
            except Exception as e:  # pragma: no cover - safety net
                eval_error = f"{type(e).__name__}: {e}"

        # 5. Persist artefacts
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        yaml_text = pipeline.to_yaml()
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        log_path = os.path.join(self.output_dir, "agent_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_name": self.data_name,
                    "task_type": ctx.task_type,
                    "success": run.success,
                    "n_turns": run.n_turns,
                    "n_errors": run.n_errors,
                    "transcript": run.transcript,
                    "ops": [s.op for s in pipeline.steps],
                    "solution_attempts": run.solution_attempts,
                    "downstream_feedback": self.downstream_feedback,
                    "downstream_eval_n": self.downstream_eval_n,
                    "max_solution_attempts": self.max_solution_attempts,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        tree_path = os.path.join(self.output_dir, "tree.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(run.tree, f, ensure_ascii=False, indent=2)

        duration = time.time() - t0
        if self.verbose:
            print("=" * 60)
            print(f"[DeepPrep] DONE in {duration:.1f}s")
            fit_str = (
                f"{eval_fitness:.4f}" if isinstance(eval_fitness, float) else "n/a"
            )
            print(f"[DeepPrep] downstream fitness = {fit_str}")
            print(f"[DeepPrep] downstream metrics = {eval_metrics}")
            if eval_error:
                print(f"[DeepPrep] eval error: {eval_error}")
            print(f"[DeepPrep] best pipeline saved to: {out_yaml_path}")
            print(f"[DeepPrep] agent log saved to:   {log_path}")
            print(f"[DeepPrep] search tree saved to: {tree_path}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": yaml_text,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": eval_fitness,
            "best_metrics": eval_metrics,
            "eval_error": eval_error,
            "agent_success": run.success,
            "agent_n_turns": run.n_turns,
            "agent_n_errors": run.n_errors,
            "agent_solution_attempts": run.solution_attempts,
            "final_pipeline_ops": [s.op for s in pipeline.steps],
            "is_legal": legal,
            "task_type": ctx.task_type,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "agent_log_path": log_path,
            "tree_path": tree_path,
        }


__all__ = ["DeepPrep"]
