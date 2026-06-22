"""Top-level CtxPipe orchestrator.

Mirrors :class:`baselines.SAGA.saga.SAGA` but uses Reinforcement Learning
(DQN) to construct a pipeline rather than genetic search.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.SAGA.pipeline import DataContext, Pipeline
from baselines.SAGA.pipeline_constraints import is_legal
from baselines.SAGA.saga import _infer_rec_context, _infer_tabular_context

from .agent import DQNAgent
from .env import PipelineEnv
from .evaluator import CtxPipeEvaluator
from .tester import CtxPipeTester
from .trainer import CtxPipeTrainer


CONFIG_KEYS = (
    "n_episodes",
    "max_steps",
    "small_n",
    "eval_full",
    "gamma",
    "eps_start",
    "eps_end",
    "eps_decay_episodes",
    "batch_size",
    "target_update_every",
    "min_buffer",
    "hidden_dim",
    "lr",
    "buffer_capacity",
    "illegal_penalty",
    "failure_reward",
    "seed",
    "fast_train",
)


class CtxPipe:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        small_n: Optional[int] = None,
        eval_full: Optional[bool] = None,
        gamma: Optional[float] = None,
        eps_start: Optional[float] = None,
        eps_end: Optional[float] = None,
        eps_decay_episodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        target_update_every: Optional[int] = None,
        min_buffer: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        lr: Optional[float] = None,
        buffer_capacity: Optional[int] = None,
        illegal_penalty: Optional[float] = None,
        failure_reward: Optional[float] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        device: str = "cpu",
        fast_train: Optional[bool] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = load_baseline_config(
            config_path or default_config_path(__file__), CONFIG_KEYS
        )
        n_episodes = resolve_config_value(cfg, "n_episodes", n_episodes)
        max_steps = resolve_config_value(cfg, "max_steps", max_steps)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        gamma = resolve_config_value(cfg, "gamma", gamma)
        eps_start = resolve_config_value(cfg, "eps_start", eps_start)
        eps_end = resolve_config_value(cfg, "eps_end", eps_end)
        eps_decay_episodes = resolve_config_value(
            cfg, "eps_decay_episodes", eps_decay_episodes
        )
        batch_size = resolve_config_value(cfg, "batch_size", batch_size)
        target_update_every = resolve_config_value(
            cfg, "target_update_every", target_update_every
        )
        min_buffer = resolve_config_value(cfg, "min_buffer", min_buffer)
        hidden_dim = resolve_config_value(cfg, "hidden_dim", hidden_dim)
        lr = resolve_config_value(cfg, "lr", lr)
        buffer_capacity = resolve_config_value(
            cfg, "buffer_capacity", buffer_capacity
        )
        illegal_penalty = resolve_config_value(
            cfg, "illegal_penalty", illegal_penalty
        )
        failure_reward = resolve_config_value(cfg, "failure_reward", failure_reward)
        seed = resolve_config_value(cfg, "seed", seed)
        fast_train = resolve_config_value(cfg, "fast_train", fast_train)

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_episodes = int(n_episodes)
        self.max_steps = int(max_steps)
        self.small_n = int(small_n) if small_n else 0
        self.eval_full = bool(eval_full)
        self.gamma = float(gamma)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.batch_size = int(batch_size)
        self.target_update_every = int(target_update_every)
        self.min_buffer = int(min_buffer)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.buffer_capacity = int(buffer_capacity)
        self.illegal_penalty = float(illegal_penalty)
        self.failure_reward = float(failure_reward)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.device = device
        self.fast_train = bool(fast_train)
        self.eps_decay_episodes = (
            int(eps_decay_episodes) if eps_decay_episodes is not None else None
        )

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "CtxPipe", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: CtxPipeEvaluator) -> DataContext:
        data = evaluator._executor._load_data()
        summary = evaluator.get_data_summary()
        if evaluator.task_type == "rec":
            return _infer_rec_context(self.data_name, summary, data)
        return _infer_tabular_context(self.data_name, summary, data)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()

        # 1. Training-time evaluator with optional small_n subsampling.
        train_evaluator = CtxPipeEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self.device,
        )
        if self.fast_train:
            train_evaluator.set_fast_mode(True)

        # 2. Build dataset context (uses the small-data view; that's fine, the
        # context vector only depends on schema/statistics).
        data_ctx = self._build_context(train_evaluator)

        if self.verbose:
            print("=" * 60)
            print(f"[CtxPipe] dataset={self.data_name}  "
                  f"task={data_ctx.task_type}")
            print(f"[CtxPipe] numeric={len(data_ctx.numeric_cols)}  "
                  f"categorical={len(data_ctx.categorical_cols)}  "
                  f"list={len(data_ctx.list_cols)}  "
                  f"text={len(data_ctx.text_cols)}")
            print(f"[CtxPipe] target={data_ctx.target_col}  "
                  f"time={data_ctx.time_col}  id={data_ctx.id_col}  "
                  f"aux={data_ctx.aux_dfs}")
            print(f"[CtxPipe] small_n={self.small_n or 'OFF'}  "
                  f"n_episodes={self.n_episodes}  "
                  f"max_steps={self.max_steps}")
            print("=" * 60)

        # 3. Build env and agent.
        env = PipelineEnv(
            evaluator=train_evaluator,
            ctx=data_ctx,
            max_steps=self.max_steps,
            illegal_penalty=self.illegal_penalty,
            failure_reward=self.failure_reward,
            seed=self.seed,
        )
        agent = DQNAgent(
            state_dim=env.state_dim,
            n_actions=env.n_actions,
            hidden=self.hidden_dim,
            lr=self.lr,
            buffer_capacity=self.buffer_capacity,
            seed=self.seed,
            device=self.device,
        )

        if self.verbose:
            print(f"[CtxPipe] state_dim={env.state_dim}  "
                  f"n_actions={env.n_actions}  "
                  f"(ops={env.n_ops}, +1 STOP)")

        # 4. Train.
        trainer = CtxPipeTrainer(
            env=env,
            agent=agent,
            n_episodes=self.n_episodes,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_decay_episodes=self.eps_decay_episodes,
            gamma=self.gamma,
            batch_size=self.batch_size,
            target_update_every=self.target_update_every,
            min_buffer=self.min_buffer,
            verbose=self.verbose,
        )
        train_out = trainer.train()

        # 5. Inference (greedy).
        tester = CtxPipeTester(verbose=self.verbose)

        # If eval_full is requested, build a separate full-data evaluator.
        full_eval: Optional[CtxPipeEvaluator] = None
        if self.eval_full:
            full_eval = CtxPipeEvaluator(
                task_dir=self.task_dir,
                data_name=self.data_name,
                data_dir=self.data_dir,
                verbose=self.verbose,
                small_n=0,
                seed=self.seed,
                device=self.device,
            )

        test_out = tester.run(agent=agent, env=env, eval_full_with=full_eval)
        final_pipeline = test_out["pipeline"]

        # 6. Decide best pipeline: prefer the *final* pipeline if it succeeds;
        #    otherwise fall back to the best successful episode.
        best_yaml: Optional[str] = None
        best_fitness: Optional[float] = None
        best_metrics: dict = {}

        best_pipeline = final_pipeline
        eval_error = test_out.get("error")

        if test_out.get("fitness") is not None:
            best_yaml = test_out["pipeline_yaml"]
            best_fitness = float(test_out["fitness"])
            best_metrics = dict(test_out.get("metrics") or {})
            best_pipeline = final_pipeline
            eval_error = None
        elif train_out.get("best_record") is not None:
            best_yaml = train_out.get("best_pipeline_yaml")
            best_fitness = train_out["best_record"].fitness
            best_metrics = dict(train_out["best_record"].metrics or {})
            if best_yaml:
                try:
                    best_pipeline = Pipeline.from_yaml(best_yaml)
                except Exception:
                    best_pipeline = final_pipeline
                    eval_error = "failed to parse fallback best_pipeline_yaml"
        else:
            best_yaml = test_out["pipeline_yaml"]
            best_fitness = None
            best_metrics = {}
            best_pipeline = final_pipeline

        # 7. Persist artefacts.
        out_yaml = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml, "w", encoding="utf-8") as f:
            f.write(best_yaml or best_pipeline.to_yaml())

        weights_path = os.path.join(self.output_dir, "q_network.pt")
        try:
            agent.save(weights_path)
        except Exception as e:
            if self.verbose:
                print(f"[CtxPipe] warning: failed to save q_network: {e}")
            weights_path = None

        duration = time.time() - t0
        if self.verbose:
            print("=" * 60)
            print(f"[CtxPipe] DONE in {duration:.1f}s")
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print(f"[CtxPipe] best fitness = {fit_str}")
            print(f"[CtxPipe] best metrics = {best_metrics}")
            print(f"[CtxPipe] best pipeline saved to: {out_yaml}")
            print(f"[CtxPipe] q-network saved to: {weights_path}")
            print(f"[CtxPipe] unique evaluations: "
                  f"{train_evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": best_yaml or best_pipeline.to_yaml(),
            "best_pipeline_path": out_yaml,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "final_pipeline_ops": [s.op for s in best_pipeline.steps],
            "is_legal": is_legal(best_pipeline, data_ctx.task_type),
            "eval_error": eval_error,
            "rl_history": [
                {
                    "episode": h.episode,
                    "reward": h.reward,
                    "success": h.success,
                    "fitness": h.fitness,
                    "ops": h.ops,
                    "n_steps": h.n_steps,
                    "duration_seconds": h.duration_seconds,
                }
                for h in train_out["history"]
            ],
            "n_unique_evaluations": train_evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "q_network_path": weights_path,
        }
