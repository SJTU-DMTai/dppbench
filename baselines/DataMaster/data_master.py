"""Top-level DataMaster orchestrator.

Mirrors the structure of :class:`baselines.DeepPrep.deepprep.DeepPrep`
(infer context -> build sandbox/evaluator/llm -> run agent -> repair ->
final downstream eval -> persist artefacts) so the CLI surface stays
consistent across the dppbench baselines.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal, repair
from baselines.common.context import _infer_rec_context, _infer_tabular_context
from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.sandbox import Sandbox

from .agent import DataMasterAgent
from .data_tree import DataTree
from .evaluator import DataMasterEvaluator
from .memory import GlobalMemory
from .scheduler import UCBScheduler, UCBSchedulerConfig


CONFIG_KEYS = (
    "llm_backend",
    "llm_model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
    "timeout",
    "max_iterations",
    "k_black",
    "max_chain_len",
    "max_depth",
    "max_err_cnt",
    "max_solution_attempts",
    "c_initial",
    "c_lower_bound",
    "decay",
    "decay_alpha",
    "decay_gamma",
    "piecewise_t1",
    "piecewise_t2",
    "reward_kind",
    "memory_top_k",
    "memory_max_chars",
    "small_n",
    "eval_full",
    "downstream_eval_n",
    "seed",
    "fast_train",
)


class DataMaster:
    """LLM-Powered tree-based agent (DataMaster paper, black-only variant)."""

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
        max_iterations: Optional[int] = None,
        k_black: Optional[int] = None,
        max_chain_len: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_err_cnt: Optional[int] = None,
        max_solution_attempts: Optional[int] = None,
        # ---- UCB scheduler ----
        c_initial: Optional[float] = None,
        c_lower_bound: Optional[float] = None,
        decay: Optional[str] = None,
        decay_alpha: Optional[float] = None,
        decay_gamma: Optional[float] = None,
        piecewise_t1: Optional[int] = None,
        piecewise_t2: Optional[int] = None,
        reward_kind: Optional[str] = None,
        # ---- Memory ----
        memory_top_k: Optional[int] = None,
        memory_max_chars: Optional[int] = None,
        # ---- Eval ----
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        downstream_eval_n: Optional[int] = None,
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
        max_iterations = resolve_config_value(cfg, "max_iterations", max_iterations)
        k_black = resolve_config_value(cfg, "k_black", k_black)
        max_chain_len = resolve_config_value(cfg, "max_chain_len", max_chain_len)
        max_depth = resolve_config_value(cfg, "max_depth", max_depth)
        max_err_cnt = resolve_config_value(cfg, "max_err_cnt", max_err_cnt)
        max_solution_attempts = resolve_config_value(
            cfg, "max_solution_attempts", max_solution_attempts
        )
        c_initial = resolve_config_value(cfg, "c_initial", c_initial)
        c_lower_bound = resolve_config_value(cfg, "c_lower_bound", c_lower_bound)
        decay = resolve_config_value(cfg, "decay", decay)
        decay_alpha = resolve_config_value(cfg, "decay_alpha", decay_alpha)
        decay_gamma = resolve_config_value(cfg, "decay_gamma", decay_gamma)
        piecewise_t1 = resolve_config_value(cfg, "piecewise_t1", piecewise_t1)
        piecewise_t2 = resolve_config_value(cfg, "piecewise_t2", piecewise_t2)
        reward_kind = resolve_config_value(cfg, "reward_kind", reward_kind)
        memory_top_k = resolve_config_value(cfg, "memory_top_k", memory_top_k)
        memory_max_chars = resolve_config_value(
            cfg, "memory_max_chars", memory_max_chars
        )
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        downstream_eval_n = resolve_config_value(
            cfg, "downstream_eval_n", downstream_eval_n
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

        self.max_iterations = int(max_iterations)
        self.k_black = int(k_black)
        self.max_chain_len = int(max_chain_len)
        self.max_depth = int(max_depth)
        self.max_err_cnt = int(max_err_cnt)
        self.max_solution_attempts = int(max_solution_attempts)

        self.scheduler_cfg = UCBSchedulerConfig(
            c_initial=float(c_initial),
            c_lower_bound=float(c_lower_bound),
            decay=str(decay),
            decay_alpha=float(decay_alpha),
            decay_gamma=float(decay_gamma),
            piecewise_t1=int(piecewise_t1),
            piecewise_t2=int(piecewise_t2),
            reward_kind=str(reward_kind),
        )

        self.memory_top_k = int(memory_top_k)
        self.memory_max_chars = int(memory_max_chars)

        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "DataMaster", data_name
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

        # Build the agent-side downstream evaluator FIRST so its
        # TrainingExecutor is decoupled from the sandbox lifecycle.
        agent_evaluator = DataMasterEvaluator(
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
                print(f"[DataMaster] dataset={self.data_name}  task={ctx.task_type}")
                print(f"[DataMaster] numeric={len(ctx.numeric_cols)}  "
                      f"categorical={len(ctx.categorical_cols)}  "
                      f"list={len(ctx.list_cols)}  text={len(ctx.text_cols)}")
                print(f"[DataMaster] target={ctx.target_col}  time={ctx.time_col}  "
                      f"id={ctx.id_col}  aux={ctx.aux_dfs}")
                print(f"[DataMaster] llm={self.llm_backend}/{self.llm_model}  "
                      f"max_iter={self.max_iterations}  k_black={self.k_black}  "
                      f"max_chain={self.max_chain_len}  max_depth={self.max_depth}")
                print(f"[DataMaster] ucb c0={self.scheduler_cfg.c_initial}  "
                      f"decay={self.scheduler_cfg.decay}  "
                      f"reward={self.scheduler_cfg.reward_kind}")
                print(f"[DataMaster] small_n={self.small_n or 'OFF'}  "
                      f"eval_full={self.eval_full}  "
                      f"downstream_eval_n={self.downstream_eval_n or 'FULL'}")
                print("=" * 60)

            # 2. Tree + Memory + Scheduler + LLM + Agent
            tree = DataTree()
            memory = GlobalMemory(
                tree=tree,
                top_k_global=self.memory_top_k,
                max_chars=self.memory_max_chars,
            )
            scheduler = UCBScheduler(self.scheduler_cfg)
            llm = self._build_llm()
            agent = DataMasterAgent(
                llm=llm,
                sandbox=sandbox,
                ctx=ctx,
                tree=tree,
                memory=memory,
                scheduler=scheduler,
                downstream_evaluator=_agent_eval_fn,
                max_iterations=self.max_iterations,
                k_black=self.k_black,
                max_chain_len=self.max_chain_len,
                max_depth=self.max_depth,
                max_err_cnt=self.max_err_cnt,
                max_solution_attempts=self.max_solution_attempts,
                seed=self.seed,
                verbose=self.verbose,
            )
            run = agent.run()
            pipeline: Pipeline = run.pipeline

            # 3. Final structural repair
            repair(pipeline, ctx.task_type, ctx)
            legal = is_legal(pipeline, ctx.task_type)

            if self.verbose:
                print(f"[DataMaster] agent done. success={run.success}  "
                      f"iterations={run.n_iterations}  "
                      f"expansions={run.n_expansions}  errors={run.n_errors}  "
                      f"steps={[s.op for s in pipeline.steps]}  legal={legal}")
        finally:
            sandbox.cleanup()

        # 4. Final downstream evaluation (NOT fed back to the agent)
        eval_metrics: dict = {}
        eval_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        if self.eval_full:
            evaluator = DataMasterEvaluator(
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
            except Exception as e:  # pragma: no cover
                eval_error = f"{type(e).__name__}: {e}"

        # 5. Persist artefacts (best_pipeline.yaml is in berka style)
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
                    "n_iterations": run.n_iterations,
                    "n_expansions": run.n_expansions,
                    "n_errors": run.n_errors,
                    "best_node_id": run.best_node_id,
                    "transcript": run.transcript,
                    "solution_attempts": run.solution_attempts,
                    "ops": [s.op for s in pipeline.steps],
                    "scheduler": {
                        "c_initial": self.scheduler_cfg.c_initial,
                        "decay": self.scheduler_cfg.decay,
                        "reward_kind": self.scheduler_cfg.reward_kind,
                    },
                    "k_black": self.k_black,
                    "max_iterations": self.max_iterations,
                    "max_chain_len": self.max_chain_len,
                    "max_depth": self.max_depth,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        tree_path = os.path.join(self.output_dir, "tree.json")
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(run.tree, f, ensure_ascii=False, indent=2)

        memory_path = os.path.join(self.output_dir, "memory.json")
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory.to_dict(), f, ensure_ascii=False, indent=2)

        duration = time.time() - t0
        if self.verbose:
            print("=" * 60)
            print(f"[DataMaster] DONE in {duration:.1f}s")
            fit_str = f"{eval_fitness:.4f}" if isinstance(eval_fitness, float) else "n/a"
            print(f"[DataMaster] downstream fitness = {fit_str}")
            print(f"[DataMaster] downstream metrics = {eval_metrics}")
            if eval_error:
                print(f"[DataMaster] eval error: {eval_error}")
            print(f"[DataMaster] best pipeline saved to: {out_yaml_path}")
            print(f"[DataMaster] agent log saved to:    {log_path}")
            print(f"[DataMaster] search tree saved to:  {tree_path}")
            print(f"[DataMaster] memory log saved to:   {memory_path}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": yaml_text,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": eval_fitness,
            "best_metrics": eval_metrics,
            "eval_error": eval_error,
            "agent_success": run.success,
            "agent_n_iterations": run.n_iterations,
            "agent_n_expansions": run.n_expansions,
            "agent_n_errors": run.n_errors,
            "agent_best_node_id": run.best_node_id,
            "agent_solution_attempts": run.solution_attempts,
            "final_pipeline_ops": [s.op for s in pipeline.steps],
            "is_legal": legal,
            "task_type": ctx.task_type,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "agent_log_path": log_path,
            "tree_path": tree_path,
            "memory_path": memory_path,
        }


__all__ = ["DataMaster"]
