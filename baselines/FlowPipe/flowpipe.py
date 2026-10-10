"""DPPBench FlowPipe adapter with task-specific policy training."""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from baselines.common.config import default_config_path, load_baseline_config, resolve_config_value
from baselines.common.pipeline import Pipeline, check_pipeline
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.state_features import primary_feature_frame
from baselines.common.utils import get_data_ctx
from baselines.common.budget import BudgetExceededError

from .agent import GFNAgent
from .context import SemanticContextLoader
from .env import FlowPipeEnv
from .trainer import FlowPipeTrainer


CONFIG_KEYS = (
    "checkpoint_path", "context_cache_dir", "llama_model_path", "llm_name", "ctx_dim",
    "hidden_dim", "learning_rate", "buffer_capacity", "failure_weight",
    "reward_temperature", "inference_samples", "inference_mode",
    "inference_top_n", "max_steps", "small_n", "eval_full", "seed",
    "fast_train",
    "train_epochs", "trajectories_per_epoch", "updates_per_epoch", "trajectory_batch_size",
)
DEFAULT_LLAMA_MODEL_PATH = "/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B"


class FlowPipe:
    """Train a fresh C-GFlowNet on the target task, then synthesize a pipeline."""

    def __init__(
        self, task_dir: str, data_name: str, data_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        context_cache_dir: Optional[str] = None,
        llama_model_path: str = DEFAULT_LLAMA_MODEL_PATH,
        llm_name: Optional[str] = None, ctx_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None, learning_rate: Optional[float] = None,
        buffer_capacity: Optional[int] = None, failure_weight: Optional[float] = None,
        reward_temperature: Optional[float] = None,
        inference_samples: Optional[int] = None, inference_mode: Optional[str] = None,
        inference_top_n: Optional[int] = None, max_steps: Optional[int] = None,
        max_pipeline_len: Optional[int] = None, small_n: Optional[int] = None,
        eval_full: Optional[bool] = None, seed: Optional[int] = None,
        fast_train: Optional[bool] = None, output_dir: Optional[str] = None,
        verbose: bool = True, device: str = "cpu", model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        train_epochs: Optional[int] = None, trajectories_per_epoch: Optional[int] = None,
        updates_per_epoch: Optional[int] = None, trajectory_batch_size: Optional[int] = None,
    ) -> None:
        cfg = load_baseline_config(config_path or default_config_path(__file__), CONFIG_KEYS)
        def value(key, explicit):
            return resolve_config_value(cfg, key, explicit)

        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.checkpoint_path = str(value("checkpoint_path", checkpoint_path) or "")
        self.context_cache_dir = str(value("context_cache_dir", context_cache_dir))
        configured_llama_path = cfg.get("llama_model_path")
        if llama_model_path == DEFAULT_LLAMA_MODEL_PATH and configured_llama_path:
            llama_model_path = configured_llama_path
        self.llama_model_path = str(llama_model_path)
        self.llm_name = str(value("llm_name", llm_name))
        self.ctx_dim = int(value("ctx_dim", ctx_dim))
        self.hidden_dim = int(value("hidden_dim", hidden_dim))
        self.learning_rate = float(value("learning_rate", learning_rate))
        self.buffer_capacity = int(value("buffer_capacity", buffer_capacity))
        self.failure_weight = float(value("failure_weight", failure_weight))
        self.reward_temperature = float(value("reward_temperature", reward_temperature))
        self.inference_samples = int(value("inference_samples", inference_samples))
        self.inference_mode = str(value("inference_mode", inference_mode))
        self.inference_top_n = int(value("inference_top_n", inference_top_n))
        configured_steps = int(value("max_steps", max_steps))
        self.max_steps = min(configured_steps, int(max_pipeline_len)) if max_pipeline_len else configured_steps
        self.small_n = int(value("small_n", small_n) or 0)
        self.eval_full = bool(value("eval_full", eval_full))
        self.seed = int(value("seed", seed))
        self.fast_train = bool(value("fast_train", fast_train))
        self.train_epochs = int(value('train_epochs', train_epochs))
        self.trajectories_per_epoch = int(value('trajectories_per_epoch', trajectories_per_epoch))
        self.updates_per_epoch = int(value('updates_per_epoch', updates_per_epoch))
        self.trajectory_batch_size = int(value('trajectory_batch_size', trajectory_batch_size))
        if min(self.train_epochs, self.trajectories_per_epoch, self.updates_per_epoch, self.trajectory_batch_size) <= 0:
            raise ValueError('FlowPipe task-training settings must be positive')
        self.output_dir = os.path.abspath(output_dir or os.path.join("outputs", "FlowPipe", data_name))
        self.verbose, self.device, self.model_name = bool(verbose), device, model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_evaluator(self) -> SearchPipelineEvaluator:
        evaluator = SearchPipelineEvaluator(
            task_dir=self.task_dir, data_name=self.data_name, data_dir=self.data_dir,
            small_n=self.small_n, seed=self.seed, device=self.device,
            model_name=self.model_name, verbose=self.verbose,
        )
        evaluator.set_evaluate_std_test(False)
        if self.fast_train:
            evaluator.set_fast_mode(True)
        return evaluator

    def precompute_context(
        self, *, device: Optional[str] = None, max_length: int = 4096,
        overwrite: bool = False,
    ) -> str:
        """Precompute and persist this task's frozen Llama embedding."""
        evaluator = self._build_evaluator()
        data = evaluator.load_data()
        if evaluator.task_type == "graph":
            raise ValueError("Official FlowPipe does not support graph tasks")
        ctx = get_data_ctx(self.data_name, data, evaluator.task_type)
        frame = (
            getattr(data, "interaction_df", None)
            if evaluator.task_type == "rec"
            else getattr(data, "train_df", None)
        )
        if frame is None:
            raise ValueError(f"[{self.data_name}] primary dataframe is unavailable")
        # Keep the target in the semantic prompt, while excluding frozen test rows.
        frame = primary_feature_frame(frame, target_col=None)
        loader = SemanticContextLoader(
            self.context_cache_dir, self.llm_name, self.ctx_dim,
            llama_model_path=self.llama_model_path,
        )
        return str(loader.precompute(
            self.data_name, frame, ctx, device=device or self.device,
            max_length=max_length, overwrite=overwrite,
        ))

    def run(self) -> dict:
        started = time.time()
        evaluator = self._build_evaluator()
        data = evaluator.load_data()
        if evaluator.task_type == "graph":
            return {
                "unsupported_task_type": "graph",
                "error": "Official FlowPipe defines tabular classification pipelines; graph tasks are unsupported.",
                "best_pipeline_yaml": None,
                "duration_seconds": time.time() - started,
            }
        ctx = get_data_ctx(self.data_name, data, evaluator.task_type)
        env = FlowPipeEnv(evaluator, ctx, max_steps=self.max_steps, seed=self.seed)
        context = SemanticContextLoader(
            self.context_cache_dir, self.llm_name, self.ctx_dim,
            llama_model_path=self.llama_model_path,
        ).load(self.data_name)
        agent = GFNAgent(
            env.state_dim, env.action_dim, hidden_dim=self.hidden_dim,
            ctx_dim=self.ctx_dim, learning_rate=self.learning_rate,
            buffer_capacity=self.buffer_capacity, failure_weight=self.failure_weight,
            reward_temperature=self.reward_temperature, device=self.device,
        )
        # Fresh task-specific policy; no cross-task checkpoint is loaded.
        agent.set_context(context)
        agent.model.train()
        trainer = FlowPipeTrainer(
            agent, [(env, context)], output_dir=self.output_dir, combos_per_epoch=1,
            trajectories_per_combo=self.trajectories_per_epoch,
            updates_per_combo=self.updates_per_epoch,
            trajectory_batch_size=self.trajectory_batch_size, seed=self.seed,
        )
        budget_error = None
        try:
            trainer.train(num_epochs=self.train_epochs)
        except BudgetExceededError as exc:
            budget_error = str(exc)
        agent.save(os.path.join(self.output_dir, 'gfn.pt'))
        self.checkpoint_path = os.path.join(self.output_dir, 'gfn.pt')
        agent.model.eval()

        candidates = list(trainer.candidates)
        count = self.inference_samples if self.inference_mode == "sample" else self.inference_top_n
        for index in range(max(1, count)):
            env.reset()
            rank = None if self.inference_mode == "sample" else index
            try:
                reward, trajectory = agent.sample_trajectory(env, greedy_rank=rank)
            except BudgetExceededError as exc:
                budget_error = str(exc)
                break
            candidates.append({
                "reward": float(reward), "pipeline": env.current_pipeline(),
                "actions": list(trajectory["actions"]), "errors": list(trajectory["errors"]),
                "logic_pipeline_id": env.logic_id,
                'success': trajectory.get('success', False),
                'fitness': trajectory.get('fitness'), 'metrics': trajectory.get('metrics', {}),
                'error_messages': trajectory.get('error_messages', []),
            })
        if not candidates:
            raise RuntimeError('FlowPipe task training produced no complete trajectory: ' + str(budget_error))
        eligible = [item for item in candidates if item.get('success')
                    and check_pipeline(item['pipeline'], ctx).valid]
        if not eligible:
            with open(os.path.join(self.output_dir, 'flowpipe_log.json'), 'w') as handle:
                json.dump({'training_mode': 'per_task', 'policy_updates': trainer.policy_updates,
                           'training_history': trainer.history, 'budget_error': budget_error,
                           'candidates': [{k: v for k, v in item.items() if k != 'pipeline'} for item in candidates]},
                          handle, indent=2, default=str)
            raise RuntimeError('FlowPipe produced no legal, successfully evaluated candidate; see flowpipe_log.json')
        best = max(eligible, key=lambda item: item["fitness"])
        pipeline: Pipeline = best["pipeline"]
        report = check_pipeline(pipeline, ctx)
        out_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(pipeline.to_yaml())
        log_path = os.path.join(self.output_dir, "flowpipe_log.json")
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump({
                "official_commit": "265a4e91af6eaaf0a734619cb71e183c0788334f",
                "checkpoint_path": self.checkpoint_path,
                "training_mode": "per_task",
                "training_history": trainer.history,
                "policy_updates": trainer.policy_updates,
                "budget_error": budget_error,
                "llama_model_path": self.llama_model_path,
                "llm_name": self.llm_name,
                "inference_mode": self.inference_mode,
                "candidates": [
                    {k: v for k, v in item.items() if k != "pipeline"} for item in candidates
                ],
            }, handle, indent=2, ensure_ascii=False)
        return {
            "best_pipeline_yaml": pipeline.to_yaml(),
            "best_pipeline_path": out_path,
            "best_fitness": best["fitness"],
            "best_metrics": best['metrics'],
            "final_pipeline_ops": [step.op for step in pipeline.steps],
            "is_legal": report.valid,
            "eval_error": None if report.valid else "; ".join(report.errors),
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "duration_seconds": time.time() - started,
            "output_dir": self.output_dir,
            "checkpoint_path": self.checkpoint_path,
            "task_checkpoint_path": os.path.join(self.output_dir, 'gfn.pt'),
            "training_stats": {"policy_updates": trainer.policy_updates, "epochs_completed": len(trainer.history)},
            "context_cache_dir": self.context_cache_dir,
            "llama_model_path": self.llama_model_path,
        }


__all__ = ["CONFIG_KEYS", "DEFAULT_LLAMA_MODEL_PATH", "FlowPipe"]
