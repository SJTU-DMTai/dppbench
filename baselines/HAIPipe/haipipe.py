"""HAIPipe baseline with frozen AIPipe inference and active HAI selection."""
from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import torch

from baselines.common.config import default_config_path, load_baseline_config, resolve_config_value
from baselines.common.pipeline import check_pipeline
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_data_ctx

from .active_learning import ActiveCandidateSelector, ScoreNN
from .agent import HAIPipeAgent
from .combiner import combine_pipelines, select_best_candidate
from .env import HAIPipeEnv
from .hi_pipeline import build_hi_pipeline


CONFIG_KEYS = (
    "max_pipeline_len", "small_n", "eval_full", "seed", "fast_train",
    "hi_source", "aipipe_checkpoint", "al_checkpoint", "al_k", "al_iterations",
    "learning_rate", "buffer_capacity",
)


class HAIPipe:
    def __init__(
        self, task_dir: str, data_name: str, data_dir: Optional[str] = None,
        max_pipeline_len: Optional[int] = None, small_n: Optional[int] = None,
        eval_full: Optional[bool] = None, seed: Optional[int] = None,
        output_dir: Optional[str] = None, verbose: bool = True, device: str = "cpu",
        fast_train: Optional[bool] = None, hi_source: Optional[str] = None,
        model_name: Optional[str] = None, config_path: Optional[str] = None,
        aipipe_checkpoint: Optional[str] = None, al_checkpoint: Optional[str] = None,
        al_k: Optional[int] = None, al_iterations: Optional[int] = None,
        **_legacy,
    ) -> None:
        cfg = load_baseline_config(config_path or default_config_path(__file__), CONFIG_KEYS)
        value = lambda key, explicit: resolve_config_value(cfg, key, explicit)
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.max_pipeline_len = int(value("max_pipeline_len", max_pipeline_len))
        self.small_n = value("small_n", small_n)
        self.eval_full = bool(value("eval_full", eval_full))
        self.seed = int(value("seed", seed))
        self.fast_train = bool(value("fast_train", fast_train))
        self.hi_source = str(value("hi_source", hi_source))
        self.aipipe_checkpoint = str(value("aipipe_checkpoint", aipipe_checkpoint))
        self.al_checkpoint = str(value("al_checkpoint", al_checkpoint))
        self.al_k = int(value("al_k", al_k))
        self.al_iterations = int(value("al_iterations", al_iterations))
        self.learning_rate = float(cfg.get("learning_rate", 1e-5))
        self.buffer_capacity = int(cfg.get("buffer_capacity", 2000))
        self.model_name, self.verbose, self.device = model_name, bool(verbose), device
        self.output_dir = output_dir or os.path.join("outputs", "HAIPipe", data_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _checkpoint(self, value: str) -> str:
        return value if os.path.isabs(value) else os.path.join(os.path.dirname(__file__), value)

    def run(self) -> dict:
        started = time.time()
        evaluator = SearchPipelineEvaluator(
            task_dir=self.task_dir, data_name=self.data_name, data_dir=self.data_dir,
            small_n=self.small_n, seed=self.seed, verbose=self.verbose,
            device=self.device, model_name=self.model_name,
        )
        evaluator.set_evaluate_std_test(False)
        if self.fast_train:
            evaluator.set_fast_mode(True)
        loaded = evaluator.load_data()
        ctx = get_data_ctx(self.data_name, loaded, evaluator.task_type)
        hi = build_hi_pipeline(
            ctx, random.Random(self.seed), task_dir=self.task_dir,
            max_len=self.max_pipeline_len, hi_source=self.hi_source,
        )
        env = HAIPipeEnv(evaluator, ctx, model_name=self.model_name, seed=self.seed)
        agent = HAIPipeAgent(
            component_actions=env.component_actions,
            logical_pipelines=env.logical_pipelines,
            primitive_vocabulary=env.primitive_vocabulary,
            model_vocabulary=env.model_vocabulary,
            data_dim=env.data_dim,
            learning_rate=self.learning_rate,
            buffer_capacity=self.buffer_capacity,
            seed=self.seed,
            device=self.device,
        )
        aipipe_path = self._checkpoint(self.aipipe_checkpoint)
        if not os.path.isfile(aipipe_path):
            raise FileNotFoundError(
                f"HAIPipe requires a cross-task pretrained AIPipe checkpoint: {aipipe_path}. "
                "Target-task online training is intentionally disabled."
            )
        policy_provenance = agent.load(aipipe_path)
        if self.data_name not in set(policy_provenance.get("held_out_tasks") or []):
            raise ValueError(
                f"HAIPipe checkpoint provenance does not declare target task {self.data_name!r} as held out"
            )
        if self.data_name in set(policy_provenance.get("training_tasks") or []):
            raise ValueError(
                f"HAIPipe checkpoint leakage: target task {self.data_name!r} was used for AIPipe pretraining"
            )
        ai, trace = env.rollout(agent, epsilon=0.0)
        candidates = combine_pipelines(
            hi, ai, ctx, random.Random(self.seed), max_len=self.max_pipeline_len,
        )
        al_model = ScoreNN(env.data_dim, len(env.primitive_vocabulary))
        al_path = self._checkpoint(self.al_checkpoint)
        if not os.path.isfile(al_path):
            raise FileNotFoundError(f"HAIPipe requires a pretrained alternative-evaluator checkpoint: {al_path}")
        al_state = torch.load(al_path, map_location="cpu")
        expected_al_metadata = {
            "data_dim": env.data_dim,
            "primitive_vocabulary": env.primitive_vocabulary,
            "sequence_length": 30,
        }
        if not isinstance(al_state, dict) or al_state.get("metadata") != expected_al_metadata:
            raise ValueError(
                "HAIPipe alternative-evaluator checkpoint is incompatible with "
                "the current DPPBench operator vocabulary"
            )
        al_provenance = dict(al_state.get("provenance") or {})
        if self.data_name not in set(al_provenance.get("held_out_tasks") or []):
            raise ValueError(
                f"HAIPipe alternative-evaluator provenance does not declare target task {self.data_name!r} as held out"
            )
        if self.data_name in set(al_provenance.get("training_tasks") or []):
            raise ValueError(
                f"HAIPipe checkpoint leakage: target task {self.data_name!r} was used for alternative-evaluator pretraining"
            )
        al_model.load_state_dict(al_state["model"])
        selector = ActiveCandidateSelector(
            al_model, k_total=self.al_k, iterations=self.al_iterations, seed=self.seed,
        )
        selected = select_best_candidate(
            evaluator, candidates, selector=selector,
            dataset_feature=env.data_feature, vocabulary=env.primitive_vocabulary,
        )
        output = selected.pipeline
        full_fitness, full_metrics = None, {}
        if self.eval_full:
            evaluator.set_small_n(0)
            if self.fast_train:
                evaluator.set_fast_mode(False)
            final = evaluator.evaluate(output)
            if final.success:
                full_fitness, full_metrics = float(final.fitness), dict(final.metrics or {})
        report = check_pipeline(output, ctx)
        yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        log_path = os.path.join(self.output_dir, "haipipe_log.json")
        with open(yaml_path, "w", encoding="utf-8") as handle:
            handle.write(output.to_yaml())
        log = {
            "data_name": self.data_name, "hi_source": self.hi_source,
            "hi_ops": hi.op_names(), "ai_ops": ai.op_names(),
            "best_pipeline_yaml": output.to_yaml(), "best_pipeline_path": yaml_path,
            "best_fitness": selected.fitness, "best_metrics": selected.metrics,
            "best_full_fitness": full_fitness, "best_full_metrics": full_metrics,
            "is_legal": report.valid, "check_errors": report.errors,
            "aipipe_trace": trace, "active_learning": selected.candidates,
            "n_hai_candidates": len(candidates),
            "n_unique_evaluations": evaluator.n_unique_evaluations,
            "duration_seconds": time.time() - started,
            "checkpoints": {"aipipe": aipipe_path, "alternative_evaluator": al_path},
        }
        with open(log_path, "w", encoding="utf-8") as handle:
            json.dump(log, handle, indent=2)
        return log


__all__ = ["HAIPipe"]
