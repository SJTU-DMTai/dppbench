"""Top-level BAT orchestrator.

Mirrors the structure of :class:`baselines.DeepPrep.deepprep.DeepPrep` and
:class:`baselines.DataMaster.data_master.DataMaster`:

  1. Build a downstream evaluator (``BATEvaluator``) and a structural
     sandbox (re-using DeepPrep's :class:`Sandbox`).
  2. Reset the sandbox and infer the dataset's :class:`DataContext` via
     SAGA's helpers (so both tabular and rec datasets are supported).
  3. Run the MCTS solver over BAT's DPAS action set; the
     :class:`BATReward` fuses the original BAT column-similarity feedback
     with a downstream LightGBM/DIN AUC signal (and optionally an
     LLM-judge term).
  4. Take the best END node, structurally repair its pipeline (mandatory
     ops, canonical ordering), and run a final full-data evaluation.
  5. Persist YAML pipeline, agent log, search tree, and best paths under
     ``outputs/BAT/<data_name>/``.
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
from baselines.SAGA.pipeline import DataContext, Pipeline
from baselines.SAGA.pipeline_constraints import is_legal, repair
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context

from baselines.DeepPrep.llm_client import LLMClient

from .evaluator import BATEvaluator
from .mcts import MCTSSolver
from .reward import BATReward, BATRewardConfig
from .sandbox import Sandbox


CONFIG_KEYS = (
    "llm_backend",
    "llm_model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
    "timeout",
    "max_rollout_steps",
    "max_depth",
    "max_chain_len",
    "exploration_constant",
    "early_stop_n_paths",
    "early_stop_eps",
    "reward_alpha",
    "reward_beta",
    "reward_gamma",
    "use_downstream",
    "use_llm_judge",
    "columns_match_threshold",
    "small_n",
    "eval_full",
    "downstream_eval_n",
    "seed",
    "fast_train",
)


class BAT:
    """LLM-driven MCTS data-preparation synthesizer."""

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
        # ---- MCTS ----
        max_rollout_steps: Optional[int] = None,
        max_depth: Optional[int] = None,
        max_chain_len: Optional[int] = None,
        exploration_constant: Optional[float] = None,
        early_stop_n_paths: Optional[int] = None,
        early_stop_eps: Optional[float] = None,
        # ---- Reward fusion ----
        reward_alpha: Optional[float] = None,
        reward_beta: Optional[float] = None,
        reward_gamma: Optional[float] = None,
        use_downstream: Optional[bool] = None,
        use_llm_judge: Optional[bool] = None,
        columns_match_threshold: Optional[float] = None,
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
        max_rollout_steps = resolve_config_value(
            cfg, "max_rollout_steps", max_rollout_steps
        )
        max_depth = resolve_config_value(cfg, "max_depth", max_depth)
        max_chain_len = resolve_config_value(cfg, "max_chain_len", max_chain_len)
        exploration_constant = resolve_config_value(
            cfg, "exploration_constant", exploration_constant
        )
        early_stop_n_paths = resolve_config_value(
            cfg, "early_stop_n_paths", early_stop_n_paths
        )
        early_stop_eps = resolve_config_value(cfg, "early_stop_eps", early_stop_eps)
        reward_alpha = resolve_config_value(cfg, "reward_alpha", reward_alpha)
        reward_beta = resolve_config_value(cfg, "reward_beta", reward_beta)
        reward_gamma = resolve_config_value(cfg, "reward_gamma", reward_gamma)
        use_downstream = resolve_config_value(cfg, "use_downstream", use_downstream)
        use_llm_judge = resolve_config_value(cfg, "use_llm_judge", use_llm_judge)
        columns_match_threshold = resolve_config_value(
            cfg, "columns_match_threshold", columns_match_threshold
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

        self.max_rollout_steps = int(max_rollout_steps)
        self.max_depth = int(max_depth)
        self.max_chain_len = int(max_chain_len)
        self.exploration_constant = float(exploration_constant)
        self.early_stop_n_paths = int(early_stop_n_paths)
        self.early_stop_eps = float(early_stop_eps)

        self.reward_cfg = BATRewardConfig(
            alpha=float(reward_alpha),
            beta=float(reward_beta),
            gamma=float(reward_gamma),
            use_downstream=bool(use_downstream),
            use_llm_judge=bool(use_llm_judge),
            columns_match_threshold=float(columns_match_threshold),
        )

        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "BAT", data_name
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

        # 1. Build the in-loop downstream evaluator (small-N).
        agent_evaluator: Optional[BATEvaluator] = None
        if self.reward_cfg.use_downstream:
            agent_evaluator = BATEvaluator(
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

        # 2. Sandbox + context inference.
        sandbox = Sandbox(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            small_n=self.small_n,
            seed=self.seed,
        )
        result_dict: dict = {}
        try:
            ctx = self._infer_context(sandbox)
            sandbox.reset()  # ensures _initial_snapshot is materialised

            if self.verbose:
                print("=" * 60)
                print(f"[BAT] dataset={self.data_name}  task={ctx.task_type}")
                print(f"[BAT] numeric={len(ctx.numeric_cols)}  "
                      f"categorical={len(ctx.categorical_cols)}  "
                      f"list={len(ctx.list_cols)}  text={len(ctx.text_cols)}")
                print(f"[BAT] target={ctx.target_col}  time={ctx.time_col}  "
                      f"id={ctx.id_col}  aux={ctx.aux_dfs}")
                print(f"[BAT] llm={self.llm_backend}/{self.llm_model}  "
                      f"max_rollout={self.max_rollout_steps}  "
                      f"max_depth={self.max_depth}  "
                      f"max_chain={self.max_chain_len}")
                print(f"[BAT] reward alpha={self.reward_cfg.alpha} "
                      f"beta={self.reward_cfg.beta} "
                      f"gamma={self.reward_cfg.gamma}  "
                      f"use_downstream={self.reward_cfg.use_downstream}  "
                      f"use_llm_judge={self.reward_cfg.use_llm_judge}")
                print(f"[BAT] small_n={self.small_n or 'OFF'}  "
                      f"eval_full={self.eval_full}  "
                      f"downstream_eval_n={self.downstream_eval_n or 'OFF'}")
                print("=" * 60)

            # 3. LLM + reward + MCTS.
            llm = self._build_llm()
            reward_model = BATReward(
                ctx=ctx,
                sandbox=sandbox,
                downstream_evaluator=agent_evaluator,
                llm=llm,
                config=self.reward_cfg,
            )
            solver = MCTSSolver(
                llm=llm,
                ctx=ctx,
                sandbox=sandbox,
                reward_model=reward_model,
                max_rollout_steps=self.max_rollout_steps,
                max_depth=self.max_depth,
                exploration_constant=self.exploration_constant,
                max_chain_len=self.max_chain_len,
                early_stop_n_paths=self.early_stop_n_paths,
                early_stop_eps=self.early_stop_eps,
                seed=self.seed,
                verbose=self.verbose,
            )
            mcts_res = solver.solve()

            best_node = mcts_res.best_node
            if best_node is None:
                # All rollouts failed -- fall back to a repair-only pipeline.
                if self.verbose:
                    print("[BAT] no successful rollout; using repair fallback.")
                pipeline = Pipeline()
            else:
                pipeline = Pipeline(steps=list(best_node.final_pipeline_steps))

            # 4. Final structural repair (canonical order + mandatory ops)
            repair(pipeline, ctx.task_type, ctx)
            legal = is_legal(pipeline, ctx.task_type)

            if self.verbose:
                print(f"[BAT] mcts done. rollouts={mcts_res.n_rollouts}  "
                      f"best_paths={len(mcts_res.best_paths)}  "
                      f"steps={[s.op for s in pipeline.steps]}  "
                      f"legal={legal}")

            # 5. Persist tree / log artefacts (before final eval).
            tree_path = os.path.join(self.output_dir, "tree.json")
            with open(tree_path, "w", encoding="utf-8") as f:
                json.dump(mcts_res.root.to_dict(), f,
                          ensure_ascii=False, indent=2)

            best_paths_path = os.path.join(self.output_dir, "best_paths.json")
            with open(best_paths_path, "w", encoding="utf-8") as f:
                json.dump(mcts_res.best_paths, f, ensure_ascii=False, indent=2)

            best_node_dump: Optional[dict] = None
            if best_node is not None:
                best_node_dump = {
                    "node_type": best_node.node_type.value,
                    "depth": best_node.depth,
                    "final_pipeline_ops": [s.op for s in best_node.final_pipeline_steps],
                    "column_similarity": best_node.column_similarity,
                    "columns_match": best_node.columns_match,
                    "downstream_fitness": best_node.downstream_fitness,
                    "downstream_metrics": best_node.downstream_metrics,
                    "exec_error": best_node.exec_error,
                    "reward_value": best_node.reward_value,
                    "reward_breakdown": best_node.reward_breakdown,
                }

            log_path = os.path.join(self.output_dir, "agent_log.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "data_name": self.data_name,
                        "task_type": ctx.task_type,
                        "n_rollouts": mcts_res.n_rollouts,
                        "n_end_nodes": len(mcts_res.end_nodes),
                        "n_best_paths": len(mcts_res.best_paths),
                        "use_downstream": self.reward_cfg.use_downstream,
                        "use_llm_judge": self.reward_cfg.use_llm_judge,
                        "reward_alpha": self.reward_cfg.alpha,
                        "reward_beta": self.reward_cfg.beta,
                        "reward_gamma": self.reward_cfg.gamma,
                        "best_node": best_node_dump,
                        "ops": [s.op for s in pipeline.steps],
                        "is_legal": legal,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        finally:
            sandbox.cleanup()

        # 6. Final downstream evaluation on the FULL data (not fed back).
        eval_metrics: dict = {}
        eval_fitness: Optional[float] = None
        eval_error: Optional[str] = None
        if self.eval_full:
            evaluator = BATEvaluator(
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
            except Exception as e:  # pragma: no cover -- safety net
                eval_error = f"{type(e).__name__}: {e}"

        # 7. Persist final pipeline YAML + report.
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        yaml_text = pipeline.to_yaml()
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)

        duration = time.time() - t0
        if self.verbose:
            print("=" * 60)
            print(f"[BAT] DONE in {duration:.1f}s")
            fit_str = (f"{eval_fitness:.4f}"
                       if isinstance(eval_fitness, float) else "n/a")
            print(f"[BAT] downstream fitness = {fit_str}")
            print(f"[BAT] downstream metrics = {eval_metrics}")
            if eval_error:
                print(f"[BAT] eval error: {eval_error}")
            print(f"[BAT] best pipeline saved to: {out_yaml_path}")
            print(f"[BAT] agent log saved to:    {log_path}")
            print(f"[BAT] search tree saved to:  {tree_path}")
            print(f"[BAT] best paths saved to:   {best_paths_path}")
            print("=" * 60)

        result_dict = {
            "best_pipeline_yaml": yaml_text,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": eval_fitness,
            "best_metrics": eval_metrics,
            "eval_error": eval_error,
            "n_rollouts": mcts_res.n_rollouts,
            "n_best_paths": len(mcts_res.best_paths),
            "final_pipeline_ops": [s.op for s in pipeline.steps],
            "is_legal": legal,
            "task_type": ctx.task_type,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "agent_log_path": log_path,
            "tree_path": tree_path,
            "best_paths_path": best_paths_path,
            "best_node": best_node_dump,
        }
        return result_dict


__all__ = ["BAT"]
