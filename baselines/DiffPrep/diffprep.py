"""Top-level DiffPrep orchestrator.

Wraps:

  * :class:`DiffPrepEvaluator` -- runs the real downstream model on a YAML
    pipeline (LightGBM for tabular, DIN for rec).
  * :class:`ContinuousPipeline` -- the differentiable pipeline parameters.
  * :class:`TabularSurrogate` / :class:`RecSurrogate` -- the inner-loop
    learner whose validation loss provides the signal for ``tau``.
  * :class:`DiffPrepTrainer` -- the bilevel optimisation loop.
  * :func:`discretize` -- argmax projection from continuous parameters to a
    legal :class:`Pipeline`.

Public entry: :class:`DiffPrep`.run() returns a dict whose schema is aligned
with :class:`baselines.CtxPipe.ctxpipe.CtxPipe`.run().
"""
from __future__ import annotations

import json
import math
import os
import random as _random
import time
import warnings
import yaml
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.config import (
    default_config_path,
    load_baseline_config,
    resolve_config_value,
)
from baselines.common.data_ctx import DataContext
from baselines.common.pipeline import Pipeline
from baselines.common.pipeline import check_pipeline
from baselines.common.utils import get_data_ctx

from .discretizer import argmax_op_names, discretize, slot_order
from .evaluator import DiffPrepEvaluator
from .search_space import ContinuousPipeline
from .slot_planner import make_slots
from .surrogate import RecSurrogate, TabularSurrogate
from .trainer import DiffPrepTrainer


CONFIG_KEYS = (
    "n_epochs",
    "small_n",
    "lr_w",
    "lr_alpha",
    "eps_finite_diff",
    "flex",
    "eval_full",
    "max_pipeline_len",
    "max_features",
    "batch_size",
    "val_ratio",
    "continuous_init_scale",
    "second_order",
    "sgd_momentum",
    "surrogate_hidden_dim",
    "rec_emb_dim",
    "gumbel_tau",
    "hard_sample",
    "seed",
    "join_overrides",
)


# ---------------------------------------------------------------------------
# Helpers: extract numeric feature tensors + labels from a loaded data object.
# ---------------------------------------------------------------------------
def _zscore(arr: np.ndarray) -> np.ndarray:
    """Column-wise z-score with safe std and clipping."""
    if arr.size == 0:
        return arr
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (arr - mu) / sd
def _tabular_feature_tensors(data, ctx: DataContext, max_features: int | None,
                             device: torch.device, fit_indices=None):
    train_df = data.train_df
    target = ctx.target_col
    if target is None or target not in train_df.columns:
        raise RuntimeError(
            "DiffPrep tabular surrogate needs a target column to train on; "
            f"none found for dataset {ctx.data_name}."
        )

    protected = {target, ctx.id_col, ctx.time_col, "__split__", "_sort_idx"}
    numeric_cols = [c for c in ctx.numeric_cols if c not in protected and c in train_df.columns]
    categorical_cols = [
        c for c in ctx.categorical_cols
        if c not in protected and c in train_df.columns
    ]
    cols = numeric_cols + categorical_cols
    if max_features:
        cols = cols[:max_features]
    if not cols:
        # Synthesise a single zero column so the surrogate can still run.
        feat = np.zeros((len(train_df), 1), dtype=np.float32)
        feature_names = ["__constant__"]
    else:
        parts = []
        feature_names = []
        for col in cols:
            if col in numeric_cols:
                values = pd.to_numeric(train_df[col], errors="coerce")
            else:
                # Official DiffPrep keeps categorical inputs and one-hot encodes
                # them. Category codes preserve that information without
                # exploding DPPBench's larger schemas during search.
                fit_values = train_df[col] if fit_indices is None else train_df[col].iloc[fit_indices]
                categories = pd.Index(fit_values.dropna().unique())
                values = pd.Series(pd.Categorical(train_df[col], categories=categories).codes).replace(-1, np.nan)
            parts.append(values.to_numpy(dtype=np.float32).reshape(-1, 1))
            feature_names.append(str(col))
        feat = np.concatenate(parts, axis=1)

    mask = np.isnan(feat).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    fit = feat if fit_indices is None else feat[fit_indices]
    mu = np.mean(fit, axis=0)
    sd = np.std(fit, axis=0)
    feat = (feat - mu) / np.where(sd < 1e-6, 1.0, sd)
    feat = np.clip(feat, -5.0, 5.0).astype(np.float32)

    regression = getattr(ctx, "prediction_task", None) == "regression"
    if regression:
        y = pd.to_numeric(train_df[target], errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("DiffPrep regression targets must be finite")
        fit_y = y if fit_indices is None else y[fit_indices]
        y = (y - fit_y.mean()) / max(float(fit_y.std()), 1e-6)
        classes = [0]
    else:
        y_codes, classes = pd.factorize(train_df[target], sort=True)
        y = y_codes.astype(np.int64)
        if (y < 0).any():
            raise ValueError("DiffPrep classification targets must not be missing")

    x_t = torch.tensor(feat, dtype=torch.float32, device=device)
    m_t = torch.tensor(mask, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32 if regression else torch.long, device=device)
    return x_t, m_t, y_t, feature_names, max(len(classes), 2)


# ---------------------------------------------------------------------------
# DiffPrep main class
# ---------------------------------------------------------------------------
class DiffPrep:
    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        n_epochs: Optional[int] = None,
        small_n: Optional[int] = None,
        lr_w: Optional[float] = None,
        lr_alpha: Optional[float] = None,
        eps_finite_diff: Optional[float] = None,
        flex: Optional[bool] = None,
        eval_full: Optional[bool] = None,
        max_pipeline_len: Optional[int] = None,
        max_features: Optional[int] = None,
        batch_size: Optional[int] = None,
        val_ratio: Optional[float] = None,
        continuous_init_scale: Optional[float] = None,
        second_order: Optional[bool] = None,
        sgd_momentum: Optional[float] = None,
        surrogate_hidden_dim: Optional[int] = None,
        rec_emb_dim: Optional[int] = None,
        gumbel_tau: Optional[float] = None,
        hard_sample: Optional[bool] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        device: str = "cpu",
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        join_overrides: Optional[dict] = None,
    ) -> None:
        path = config_path or default_config_path(__file__)
        # New optional key must not invalidate existing server config files.
        with open(path, encoding='utf-8') as f:
            declared = yaml.safe_load(f)
        keys = CONFIG_KEYS if isinstance(declared, dict) and 'join_overrides' in declared else tuple(
            k for k in CONFIG_KEYS if k != 'join_overrides')
        cfg = load_baseline_config(path, keys)
        cfg.setdefault('join_overrides', {})
        n_epochs = resolve_config_value(cfg, "n_epochs", n_epochs)
        small_n = resolve_config_value(cfg, "small_n", small_n)
        lr_w = resolve_config_value(cfg, "lr_w", lr_w)
        lr_alpha = resolve_config_value(cfg, "lr_alpha", lr_alpha)
        eps_finite_diff = resolve_config_value(
            cfg, "eps_finite_diff", eps_finite_diff
        )
        flex = resolve_config_value(cfg, "flex", flex)
        eval_full = resolve_config_value(cfg, "eval_full", eval_full)
        max_pipeline_len = resolve_config_value(
            cfg, "max_pipeline_len", max_pipeline_len
        )
        max_features = resolve_config_value(cfg, "max_features", max_features)
        batch_size = resolve_config_value(cfg, "batch_size", batch_size)
        val_ratio = resolve_config_value(cfg, "val_ratio", val_ratio)
        continuous_init_scale = resolve_config_value(
            cfg, "continuous_init_scale", continuous_init_scale
        )
        second_order = resolve_config_value(cfg, "second_order", second_order)
        sgd_momentum = resolve_config_value(cfg, "sgd_momentum", sgd_momentum)
        surrogate_hidden_dim = resolve_config_value(
            cfg, "surrogate_hidden_dim", surrogate_hidden_dim
        )
        rec_emb_dim = resolve_config_value(cfg, "rec_emb_dim", rec_emb_dim)
        gumbel_tau = resolve_config_value(cfg, "gumbel_tau", gumbel_tau)
        hard_sample = resolve_config_value(cfg, "hard_sample", hard_sample)
        seed = resolve_config_value(cfg, "seed", seed)
        self.join_overrides = resolve_config_value(cfg, 'join_overrides', join_overrides)
        if not isinstance(self.join_overrides, dict):
            raise ValueError('DiffPrep join_overrides must map dataset names to join configurations')

        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.n_epochs = int(n_epochs)
        self.small_n = int(small_n) if small_n else 0
        self.lr_w = float(lr_w)
        self.lr_alpha = float(lr_alpha)
        self.eps_finite_diff = float(eps_finite_diff)
        self.flex = bool(flex)
        self.eval_full = bool(eval_full)
        self.max_pipeline_len = int(max_pipeline_len) if max_pipeline_len else None
        self.max_features = int(max_features) if max_features else None
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.continuous_init_scale = float(continuous_init_scale)
        self.second_order = bool(second_order)
        self.sgd_momentum = float(sgd_momentum)
        self.surrogate_hidden_dim = int(surrogate_hidden_dim)
        self.rec_emb_dim = int(rec_emb_dim)
        self.gumbel_tau = float(gumbel_tau)
        self.hard_sample = bool(hard_sample)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.model_name = model_name

        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "DiffPrep", data_name
        )
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device(device)  # honours --gpu_id; falls back to CPU
        self._device_str = device

    # ------------------------------------------------------------------
    def _build_context(self, evaluator: DiffPrepEvaluator) -> Tuple[DataContext, object]:
        data = evaluator.load_data()
        return get_data_ctx(self.data_name, data, evaluator.task_type), data

    # ------------------------------------------------------------------
    def run(self) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        _random.seed(self.seed)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        t0 = time.time()

        # 1) train-time evaluator with subsampling
        train_evaluator = DiffPrepEvaluator(
            task_dir=self.task_dir,
            data_name=self.data_name,
            data_dir=self.data_dir,
            verbose=self.verbose,
            small_n=self.small_n,
            seed=self.seed,
            device=self._device_str,
            model_name=self.model_name,
        )
        train_evaluator.set_evaluate_std_test(False)
        ctx, data = self._build_context(train_evaluator)
        task_type = ctx.task_type
        structural_steps = []
        join_config = self.join_overrides.get(self.data_name)
        if join_config is not None and task_type != 'tabular':
            raise ValueError('Explicit DiffPrep joins currently support tabular tasks only')
        if task_type == "tabular":
            from .data_protocol import prepare_tabular_search
            splits, ctx, structural_steps = prepare_tabular_search(
                data, train_evaluator._executor.get_model_config(), ctx,
                self.seed, self.small_n, join_config=join_config,
            )
        elif task_type == "rec":
            from .rec_search import prepare_rec_search
            ctx, structural_steps, rec_audit = prepare_rec_search(
                data, train_evaluator._executor.get_model_config(), self.seed, self.small_n)
            ctx.data_name = self.data_name

        if self.verbose:
            print("=" * 60)
            print(f"[DiffPrep] dataset={self.data_name}  task={task_type}")
            print(f"[DiffPrep] numeric={len(ctx.numeric_cols)}  "
                  f"categorical={len(ctx.categorical_cols)}")
            print(f"[DiffPrep] target={ctx.target_col}  time={ctx.time_col}  "
                  f"id={ctx.id_col}")
            print(f"[DiffPrep] flex={self.flex}  small_n={self.small_n or 'OFF'}  "
                  f"n_epochs={self.n_epochs}")
            print("=" * 60)

        # 2) Build slots
        slots = make_slots(task_type, ctx, max_pipeline_len=self.max_pipeline_len)
        if self.verbose:
            print(f"[DiffPrep] {len(slots)} slots: "
                  + ", ".join(f"{s.category.value}({s.kind},m={s.n_candidates})" for s in slots))

        # 3) Build surrogate tensors/model.  The continuous pipeline is created
        # after feature extraction so its architecture logits can be feature-wise,
        # matching the original DiffPrep relaxation.
        if task_type == "graph":
            from .graph_adapter import graph_search_inputs
            feat, mask, feature_names, surrogate, train_step, val_step = graph_search_inputs(
                data, self.max_features, self.device, self.surrogate_hidden_dim,
                self.gumbel_tau, self.hard_sample,
            )
        elif task_type == "rec":
            from .rec_search import RecTransformBank, fixed_rec_list_inputs
            from baselines.common.pipeline import PipelineStep
            import copy
            rec_bank = RecTransformBank(data, ctx, slots, self.seed, flex=self.flex, max_features=self.max_features)
            # Canonical train-fitted discrete IDs and deployed item history.
            data.interaction_df = rec_bank.frame.copy()
            data._remap_ids()
            frame = data.interaction_df
            target = ctx.target_col
            labels = pd.to_numeric(frame[target], errors="raise")
            if labels.isna().any() or not labels.isin([0, 1]).all():
                raise ValueError("Frozen recommendation labels must be binary and non-null")
            feat = torch.arange(len(frame), device=self.device, dtype=torch.long).reshape(-1, 1)
            mask = torch.zeros_like(feat)
            u_ids = torch.tensor(frame[ctx.user_col].to_numpy(), dtype=torch.long, device=self.device)
            i_ids = torch.tensor(frame[ctx.item_col].to_numpy(), dtype=torch.long, device=self.device)
            y = torch.tensor(labels.to_numpy(dtype=np.float32), device=self.device)
            from dppbench.models.inputs import pad_sequences
            history_array, history_length_array = pad_sequences(frame['item_id_seq'].tolist(),
                data.model_cfg.get('feature', {}).get('seq_max_len', 20))
            histories = torch.tensor(history_array, dtype=torch.long, device=self.device)
            history_lengths = torch.tensor(history_length_array, dtype=torch.long, device=self.device)
            history_mask = torch.arange(histories.size(1), device=self.device).unsqueeze(0) < history_lengths.unsqueeze(1)
            list_specs, fixed_lists = fixed_rec_list_inputs(data, self.device)
            rec_audit['fixed_list_features'] = list_specs
            surrogate = RecSurrogate(
                n_users=max(len(data.category_maps_.get(ctx.user_col, {}))+1, 2),
                n_items=max(len(data.category_maps_.get(ctx.item_col, {}))+1, 2),
                feat_dim=rec_bank.output_dim, emb_dim=self.rec_emb_dim,
                hidden=self.surrogate_hidden_dim, history=True,
                list_feature_specs=list_specs).to(self.device)
            tr_idx = np.flatnonzero(frame['__split__'].eq('train').to_numpy())
            va_idx = np.flatnonzero(frame['__split__'].eq('val').to_numpy())
            tr_idx_t = torch.tensor(tr_idx, dtype=torch.long, device=self.device)
            va_idx_t = torch.tensor(va_idx, dtype=torch.long, device=self.device)

            from baselines.common.search_cost import record_work

            def rec_loss(continuous, model, indices, partition):
                sel = indices[torch.randperm(len(indices), device=self.device)[:min(self.batch_size, len(indices))]]
                transformed = continuous(feat[sel], mask[sel],
                    gumbel_tau=self.gumbel_tau, hard_sample=self.hard_sample)
                batch_lists = {name: {key: tensor[sel] for key, tensor in values.items()}
                               for name, values in fixed_lists.items()}
                logits = model(u_ids[sel], i_ids[sel], transformed, histories[sel],
                               history_lengths[sel], history_mask[sel], batch_lists)
                loss = F.binary_cross_entropy_with_logits(logits, y[sel])
                record_work('proxy_work.' + partition, supervised_rows=len(sel), forwarded_rows=len(sel))
                acc = ((torch.sigmoid(logits) > .5).float() == y[sel]).float().mean().item()
                return loss, acc

            train_step = lambda c, m: rec_loss(c, m, tr_idx_t, 'train')[0]
            val_step = lambda c, m: rec_loss(c, m, va_idx_t, 'val')
        else:
            # Use the task's frozen training pool and its own split protocol.
            # In particular, time-series validation must remain chronological.
            train_frame, val_frame = splits["train"], splits["val"]
            if train_frame.empty or val_frame.empty:
                raise ValueError("DiffPrep requires nonempty training and validation partitions")
            data.train_df = pd.concat([train_frame, val_frame], ignore_index=True)
            tr_idx = np.arange(len(train_frame))
            va_idx = np.arange(len(train_frame), len(data.train_df))
            regression = getattr(ctx, "prediction_task", None) == "regression"
            feat, mask, y, feature_names, n_classes = _tabular_feature_tensors(
                data, ctx, self.max_features, self.device, fit_indices=tr_idx,
            )
            surrogate = TabularSurrogate(
                in_dim=feat.shape[1], n_classes=n_classes, regression=regression,
            ).to(self.device)
            tr_idx_t = torch.tensor(tr_idx, dtype=torch.long, device=self.device)
            va_idx_t = torch.tensor(va_idx, dtype=torch.long, device=self.device)

            from baselines.common.search_cost import record_work

            def train_step(continuous: ContinuousPipeline, model: nn.Module):
                n = tr_idx_t.numel()
                if n == 0:
                    return torch.zeros((), device=self.device, requires_grad=True)
                bsz = min(self.batch_size, n)
                sel = tr_idx_t[torch.randperm(n, device=self.device)[:bsz]]
                fx = feat[sel]
                mx = mask[sel]
                yy = y[sel]
                out = continuous(
                    fx, mx,
                    gumbel_tau=self.gumbel_tau,
                    hard_sample=self.hard_sample,
                )
                logits = model(out)
                loss = F.mse_loss(logits.squeeze(-1), yy) if regression else F.cross_entropy(logits, yy)
                record_work('proxy_work.train', supervised_rows=len(sel), forwarded_rows=len(sel))
                return loss

            def val_step(continuous: ContinuousPipeline, model: nn.Module):
                n = va_idx_t.numel()
                if n == 0:
                    z = torch.zeros((), device=self.device, requires_grad=True)
                    return z, 0.0
                bsz = min(self.batch_size, n)
                sel = va_idx_t[torch.randperm(n, device=self.device)[:bsz]]
                fx = feat[sel]
                mx = mask[sel]
                yy = y[sel]
                out = continuous(
                    fx, mx,
                    gumbel_tau=self.gumbel_tau,
                    hard_sample=self.hard_sample,
                )
                logits = model(out)
                loss = F.mse_loss(logits.squeeze(-1), yy) if regression else F.cross_entropy(logits, yy)
                record_work('proxy_work.val', supervised_rows=len(sel), forwarded_rows=len(sel))
                with torch.no_grad():
                    if regression:
                        acc = -float(torch.sqrt(loss).item())
                    else:
                        pred = logits.argmax(dim=1)
                        acc = (pred == yy).float().mean().item()
                return loss, acc

        cont = ContinuousPipeline(
            slots,
            flex=self.flex,
            init_scale=self.continuous_init_scale,
            n_features=(len(rec_bank.feature_names) if task_type == "rec" else int(feat.shape[1])),
            feature_names=(rec_bank.feature_names if task_type == "rec" else feature_names),
        ).to(self.device)

        if task_type == "rec":
            cont.rec_bank = rec_bank
            cont.register_buffer('rec_candidate_mask', torch.tensor(rec_bank.valid_mask, device=self.device))
            cont.rec_context, cont.rec_seed, cont.rec_data = ctx, self.seed, data
            cont.rec_sequence_step = copy.deepcopy(data._diffprep_sequence_step)

        # 4) Bilevel training
        trainer = DiffPrepTrainer(
            continuous=cont,
            surrogate=surrogate,
            train_step_fn=train_step,
            val_step_fn=val_step,
            argmax_op_fn=argmax_op_names,
            n_epochs=self.n_epochs,
            lr_w=self.lr_w,
            lr_alpha=self.lr_alpha,
            eps_finite_diff=self.eps_finite_diff,
            second_order=self.second_order,
            sgd_momentum=self.sgd_momentum,
            verbose=self.verbose,
        )
        history = trainer.train()

        # 5) Discretise
        rng = _random.Random(self.seed)
        pipe = discretize(cont, slots, ctx, rng=rng, structural_steps=structural_steps)
        if task_type == 'tabular':
            from .data_protocol import assert_join_replay
            assert_join_replay(structural_steps, pipe)
        pipe.metadata['diffprep_search_contract'] = {
            'version': 5 if task_type == 'rec' else 3,
            'structural_operations': 'fixed_seeded_adapters_not_differentiably_optimized',
            'structural_replay': 'same_join_params_in_search_and_export',
            'search_joins': [s.params for s in structural_steps],
            'search_feature_columns': list(cont.feature_names),
            'exported_joins': [s.params for s in pipe.steps if s.op == 'JoinTable'],
        }
        if task_type == 'rec':
            pipe.metadata['diffprep_search_contract'].update(rec_audit, representation=rec_bank.contract(),
                search_joins=[step.params for step in structural_steps], join_replay_checked=True)
        if task_type == 'tabular':
            pipe.metadata['diffprep_search_contract']['join_selection'] = data._diffprep_join_selection
            pipe.metadata['diffprep_search_contract']['join_replay_checked'] = True
        discrete_slot_order = slot_order(cont)
        check_report = check_pipeline(pipe, ctx)
        if not check_report.valid:
            raise RuntimeError(
                "DiffPrep produced an invalid pipeline: "
                + "; ".join(check_report.errors)
            )

        # 6) Persist artefacts
        out_yaml_path = os.path.join(self.output_dir, "best_pipeline.yaml")
        with open(out_yaml_path, "w", encoding="utf-8") as f:
            f.write(pipe.to_yaml())
        # The common Pipeline YAML does not serialize arbitrary metadata.
        # Persist the audit explicitly rather than silently losing it at export.
        search_contract = pipe.metadata['diffprep_search_contract']
        with open(os.path.join(self.output_dir, 'search_contract.json'), 'w', encoding='utf-8') as f:
            json.dump(search_contract, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print('[DiffPrep search contract] ' + json.dumps(search_contract, ensure_ascii=False))

        weights_path = os.path.join(self.output_dir, "pipeline_weights.pt")
        try:
            torch.save({
                "tau": cont.tau.detach().cpu(),
                "theta": cont.theta.detach().cpu() if cont.theta is not None else None,
                "alpha": cont.alpha().detach().cpu(),
                "discrete_slot_order": discrete_slot_order,
                "best_epoch": trainer.best_epoch,
                "surrogate": surrogate.state_dict(),
            }, weights_path)
        except Exception as e:
            if self.verbose:
                print(f"[DiffPrep] warning: failed to save pipeline weights: {e}")
            weights_path = None

        history_path = os.path.join(self.output_dir, "search_history.json")
        try:
            trainer.save_history(history_path)
        except Exception as e:
            if self.verbose:
                print(f"[DiffPrep] warning: failed to save search history: {e}")

        # 7) Final evaluation on real downstream model
        best_fitness: Optional[float] = None
        best_metrics: dict = {}
        eval_error: Optional[str] = None
        if self.eval_full:
            try:
                full_eval = DiffPrepEvaluator(
                    task_dir=self.task_dir,
                    data_name=self.data_name,
                    data_dir=self.data_dir,
                    verbose=self.verbose,
                    small_n=0,
                    seed=self.seed,
                    device=self._device_str,
                    model_name=self.model_name,
                )
                full_eval.set_evaluate_std_test(False)
                ev = full_eval.evaluate(pipe)
                if ev.success:
                    best_fitness = float(ev.fitness)
                    best_metrics = dict(ev.metrics or {})
                else:
                    eval_error = ev.error
                    if self.verbose:
                        print(f"[DiffPrep] full eval failed: {ev.error}")
            except Exception as e:
                eval_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[DiffPrep] full eval raised: {e}")

        duration = time.time() - t0

        if self.verbose:
            print("=" * 60)
            print(f"[DiffPrep] DONE in {duration:.1f}s")
            fit_str = f"{best_fitness:.4f}" if isinstance(best_fitness, float) else "n/a"
            print(f"[DiffPrep] best fitness = {fit_str}")
            print(f"[DiffPrep] best metrics = {best_metrics}")
            print(f"[DiffPrep] best pipeline saved to: {out_yaml_path}")
            print(f"[DiffPrep] weights saved to: {weights_path}")
            print(f"[DiffPrep] history saved to: {history_path}")
            print(f"[DiffPrep] unique evaluations: "
                  f"{train_evaluator.n_unique_evaluations}")
            print("=" * 60)

        return {
            "best_pipeline_yaml": pipe.to_yaml(),
            "search_contract": search_contract,
            "best_pipeline_path": out_yaml_path,
            "best_fitness": best_fitness,
            "best_metrics": best_metrics,
            "eval_error": eval_error,
            "is_legal": check_report.valid,
            "check_errors": list(check_report.errors),
            "check_warnings": list(check_report.warnings),
            "final_pipeline_ops": [s.op for s in pipe.steps],
            "flex_enabled": bool(cont.flex),
            "alpha": cont.alpha().detach().cpu().tolist(),
            "discrete_slot_order": discrete_slot_order,
            "search_history": [
                {
                    "epoch": rec.epoch,
                    "train_loss": rec.train_loss,
                    "val_loss": rec.val_loss,
                    "val_acc": rec.val_acc,
                    "argmax_pipeline": rec.argmax_pipeline,
                    "duration_seconds": rec.duration_seconds,
                }
                for rec in history
            ],
            "best_epoch": trainer.best_epoch,
            "best_val_loss": trainer.best_val_loss,
            "n_unique_evaluations": train_evaluator.n_unique_evaluations,
            "duration_seconds": duration,
            "output_dir": self.output_dir,
            "weights_path": weights_path,
        }


__all__ = ["DiffPrep"]
