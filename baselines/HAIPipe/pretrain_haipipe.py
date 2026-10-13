#!/usr/bin/env python
"""Pretrain DPPBench HAIPipe checkpoints on an explicit task split."""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.HAIPipe.active_learning import ScoreNN, pipeline_features, pretrain_score_model
from baselines.HAIPipe.combiner import combine_pipelines
from baselines.HAIPipe.env import HAIPipeEnv
from baselines.HAIPipe.haipipe import CONFIG_KEYS
from baselines.common.config import load_baseline_config
from baselines.HAIPipe.hi_pipeline import build_hi_pipeline
from baselines.HAIPipe.pretrain import CrossTaskPretrainer, save_alternative_evaluator
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_data_ctx


def parse_args():
    config = load_baseline_config(os.path.join(ROOT, "baselines", "HAIPipe", "config.yaml"), CONFIG_KEYS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_names", nargs="+", required=True,
                        help="Explicit training tasks; target/test tasks must not be listed")
    parser.add_argument("--model", required=True)
    parser.add_argument("--held_out_data_names", nargs="+", required=True,
                        help="Target/test tasks recorded in provenance and forbidden from training")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--frames", type=int, default=config["frames"])
    parser.add_argument("--small_n", type=int, default=3000)
    parser.add_argument("--max_pipeline_len", type=int, default=8)
    parser.add_argument("--al_candidates_per_task", type=int, default=40)
    parser.add_argument("--al_epochs", type=int, default=config["al_epochs"])
    for key in ("batch_size", "logic_batch_size", "buffer_capacity", "train_interval", "al_batch_size"):
        parser.add_argument(f"--{key}", type=int, default=config[key])
    for key in ("learning_rate", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay", "al_learning_rate"):
        parser.add_argument(f"--{key}", type=float, default=config[key])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    overlap = sorted(set(args.data_names) & set(args.held_out_data_names))
    if overlap:
        raise ValueError(f"HAIPipe pretraining/test task leakage: {overlap}")
    device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"
    run_dir = os.path.join(os.path.abspath(args.output_dir), "HAIPipe", "pretrain", args.model)
    os.makedirs(run_dir, exist_ok=True)
    environments = []
    contexts = []
    for task in args.data_names:
        task_dir = os.path.join(ROOT, "dppbench", "tasks", task)
        evaluator = SearchPipelineEvaluator(
            task_dir=task_dir, data_name=task,
            data_dir=os.path.join(os.path.abspath(args.data_dir), task, "data"),
            small_n=args.small_n, seed=args.seed, device=device,
            model_name=args.model, verbose=True,
        )
        evaluator.set_evaluate_std_test(False)
        data = evaluator.load_data()
        ctx = get_data_ctx(task, data, evaluator.task_type)
        environments.append(HAIPipeEnv(
            evaluator, ctx, model_name=args.model, seed=args.seed,
        ))
        contexts.append(ctx)

    aipipe_path = os.path.join(run_dir, "aipipe.pt")
    trainer = CrossTaskPretrainer(
        environments, frames=args.frames, seed=args.seed, device=device,
        learning_rate=args.learning_rate, buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size, logic_batch_size=args.logic_batch_size,
        gamma=args.gamma, train_interval=args.train_interval,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
    )
    policy_result = trainer.train(
        aipipe_path, held_out_tasks=args.held_out_data_names,
    )

    rng = random.Random(args.seed)
    dataset_rows = []
    pipeline_rows = []
    labels = []
    for task, env, ctx in zip(args.data_names, environments, contexts):
        hi = build_hi_pipeline(
            ctx, rng, task_dir=os.path.join(ROOT, "dppbench", "tasks", task),
            max_len=args.max_pipeline_len, hi_source="prepare_yaml",
        )
        ai, _ = env.rollout(trainer.agent, epsilon=0.0)
        candidates = combine_pipelines(
            hi, ai, ctx, rng, max_len=args.max_pipeline_len,
        )
        if len(candidates) > args.al_candidates_per_task:
            candidates = rng.sample(candidates, args.al_candidates_per_task)
        task_samples = []
        for candidate in candidates:
            result = env.evaluator.evaluate(candidate)
            if result.success:
                task_samples.append((candidate, float(result.fitness)))
        if not task_samples:
            continue
        mean = sum(score for _, score in task_samples) / len(task_samples)
        for candidate, score in task_samples:
            dataset_rows.append(env.data_feature)
            pipeline_rows.append(pipeline_features(candidate, env.primitive_vocabulary))
            labels.append(score - mean)

    if len(labels) < 2:
        raise RuntimeError("Not enough successful HAI candidates to pretrain alternative evaluator")
    model = ScoreNN(environments[0].data_dim, len(environments[0].primitive_vocabulary)).to(device)
    loss = pretrain_score_model(
        model, np.stack(dataset_rows), np.stack(pipeline_rows), np.asarray(labels),
        epochs=args.al_epochs, learning_rate=args.al_learning_rate, batch_size=args.al_batch_size,
    )
    al_path = os.path.join(run_dir, "alternative_evaluator.pt")
    save_alternative_evaluator(
        model, al_path, data_dim=environments[0].data_dim,
        primitive_vocabulary=environments[0].primitive_vocabulary,
        provenance={
            "training_tasks": args.data_names,
            "held_out_tasks": args.held_out_data_names,
            "training_samples": len(labels),
        },
    )
    print({
        "policy": policy_result,
        "alternative_evaluator": al_path,
        "al_training_samples": len(labels),
        "al_final_loss": loss[-1],
        "training_tasks": args.data_names,
    })


if __name__ == "__main__":
    main()
