"""DiffPrep bilevel trainer.

This follows the DiffPrep/DARTS training structure: architecture parameters are
updated from validation loss after a virtual model-weight step on training loss;
model weights are then updated on training loss.  When ``second_order=True`` the
architecture gradient includes the finite-difference Hessian-vector correction
used by the original DiffPrep implementation.
"""
from __future__ import annotations

import json
import os
import time
import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
import torch.nn as nn

from .search_space import ContinuousPipeline
from baselines.common.search_cost import measured_call


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    argmax_pipeline: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class DiffPrepTrainer:
    """Bilevel optimiser for the continuous pipeline + surrogate model."""

    def __init__(
        self,
        continuous: ContinuousPipeline,
        surrogate: nn.Module,
        train_step_fn: Callable[[ContinuousPipeline, nn.Module], torch.Tensor],
        val_step_fn: Callable[[ContinuousPipeline, nn.Module], Tuple[torch.Tensor, float]],
        argmax_op_fn: Callable[[ContinuousPipeline], List[str]],
        n_epochs: int = 5,
        lr_w: float = 1e-2,
        lr_alpha: float = 1e-3,
        eps_finite_diff: float = 1e-3,
        second_order: bool = False,
        sgd_momentum: float = 0.9,
        arch_betas: tuple[float, float] = (0.5, 0.999),
        verbose: bool = True,
    ) -> None:
        self.continuous = continuous
        self.surrogate = surrogate
        self.train_step_fn = train_step_fn
        self.val_step_fn = val_step_fn
        self.argmax_op_fn = argmax_op_fn
        self.n_epochs = int(n_epochs)
        self.lr_w = float(lr_w)
        self.lr_alpha = float(lr_alpha)
        self.eps = float(eps_finite_diff)
        self.second_order = bool(second_order)
        self.sgd_momentum = float(sgd_momentum)
        self.verbose = bool(verbose)

        self.opt_w = torch.optim.SGD(
            self.surrogate.parameters(), lr=self.lr_w, momentum=self.sgd_momentum
        )
        self.arch_params = [self.continuous.tau]
        if self.continuous.theta is not None:
            self.arch_params.append(self.continuous.theta)
        self.opt_alpha = torch.optim.Adam(
            self.arch_params, lr=self.lr_alpha, betas=arch_betas
        )
        self.history: List[EpochRecord] = []
        self.best_epoch: int | None = None
        self.best_val_loss = float("inf")

    @staticmethod
    def _params(module: nn.Module) -> list[torch.nn.Parameter]:
        return [p for p in module.parameters() if p.requires_grad]

    @staticmethod
    def _zeros_like(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
        return [torch.zeros_like(p) for p in params]

    @staticmethod
    def _grad_or_zero(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
        grads = measured_call('gradient.autograd', torch.autograd.grad,
                              loss, params, allow_unused=True, retain_graph=True)
        return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]

    def _virtual_model_step(self) -> tuple[list[torch.Tensor], float]:
        model_params = self._params(self.surrogate)
        backup = [p.detach().clone() for p in model_params]
        train_loss = measured_call('proxy_train.virtual', self.train_step_fn,
                                   self.continuous, self.surrogate)
        grads_w = self._grad_or_zero(train_loss, model_params)
        with torch.no_grad():
            for p, g in zip(model_params, grads_w):
                p.sub_(self.lr_w * g)
        return backup, float(train_loss.detach().cpu().item())

    def _restore_model(self, backup: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, old in zip(self._params(self.surrogate), backup):
                p.copy_(old)

    def _hessian_vector_product(
        self,
        backup: list[torch.Tensor],
        vector: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        model_params = self._params(self.surrogate)
        norm = torch.sqrt(sum((v.detach() ** 2).sum() for v in vector)).clamp(min=1e-12)
        weight_norm = torch.sqrt(
            sum((p.detach() ** 2).sum() for p in model_params)
        ).clamp(min=1e-12)
        # Match the official implementation:
        #   eps * ||w|| / ||dL_val/dw||
        r = self.eps * weight_norm / norm

        # Centered differences must evaluate the SAME stochastic function.
        # Replay sampling/dropout for -r, then advance the random stream once.
        def capture_rng():
            return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

        def restore_rng(state):
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            if state[3] is not None:
                torch.cuda.set_rng_state_all(state[3])

        before_rng = capture_rng()
        after_rng = before_rng
        buffers = [(b, b.detach().clone()) for b in self.surrogate.buffers()]
        try:
            with torch.no_grad():
                for p, old, v in zip(model_params, backup, vector):
                    p.copy_(old + r * v)
            loss_pos = measured_call('proxy_train.hvp_positive', self.train_step_fn,
                                     self.continuous, self.surrogate)
            grad_pos = self._grad_or_zero(loss_pos, self.arch_params)
            after_rng = capture_rng()
            restore_rng(before_rng)
            with torch.no_grad():
                for b, old in buffers:
                    b.copy_(old)
                for p, old, v in zip(model_params, backup, vector):
                    p.copy_(old - r * v)
            loss_neg = measured_call('proxy_train.hvp_negative', self.train_step_fn,
                                     self.continuous, self.surrogate)
            grad_neg = self._grad_or_zero(loss_neg, self.arch_params)
        finally:
            self._restore_model(backup)
            with torch.no_grad():
                for b, old in buffers:
                    b.copy_(old)
            restore_rng(after_rng)
        return [(gp - gn) / (2.0 * r) for gp, gn in zip(grad_pos, grad_neg)]

    def _step_arch(self) -> Tuple[float, float, float]:
        """One outer architecture update plus virtual train-loss value."""
        self.opt_alpha.zero_grad()
        model_params = self._params(self.surrogate)
        backup, virtual_train_loss = self._virtual_model_step()

        val_loss, val_acc = measured_call('proxy_val.architecture', self.val_step_fn,
                                         self.continuous, self.surrogate)
        dval_dalpha = self._grad_or_zero(val_loss, self.arch_params)

        if self.second_order:
            dval_dw = self._grad_or_zero(val_loss, model_params)
            hessian = self._hessian_vector_product(backup, dval_dw)
            arch_grads = [g - self.lr_w * h for g, h in zip(dval_dalpha, hessian)]
        else:
            arch_grads = dval_dalpha
            self._restore_model(backup)

        for p, g in zip(self.arch_params, arch_grads):
            p.grad = g.detach().clone()
        measured_call('optimizer.architecture', self.opt_alpha.step)
        return (
            float(val_loss.detach().cpu().item()),
            float(val_acc),
            virtual_train_loss,
        )

    def _step_inner(self) -> float:
        self.opt_w.zero_grad()
        train_loss = measured_call('proxy_train.inner', self.train_step_fn,
                                   self.continuous, self.surrogate)
        measured_call('gradient.backward', train_loss.backward)
        measured_call('optimizer.weights', self.opt_w.step)
        return float(train_loss.detach().cpu().item())

    def train(self) -> List[EpochRecord]:
        self.history.clear()
        self.best_epoch = None
        self.best_val_loss = float("inf")
        best_continuous = None
        best_surrogate = None
        self.continuous.train()
        self.surrogate.train()

        for epoch in range(self.n_epochs):
            t0 = time.time()
            val_loss, val_acc, virtual_train_loss = self._step_arch()
            train_loss = self._step_inner()

            argmax_ops = self.argmax_op_fn(self.continuous)
            rec = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                argmax_pipeline=argmax_ops,
                duration_seconds=time.time() - t0,
            )
            self.history.append(rec)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_continuous = deepcopy(self.continuous.state_dict())
                best_surrogate = deepcopy(self.surrogate.state_dict())
            if self.verbose:
                print(
                    f"[DiffPrep] epoch={epoch+1}/{self.n_epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"virtual_train={virtual_train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                    f"ops={argmax_ops}"
                )
        # The official experiment discretizes/evaluates the checkpoint with the
        # smallest validation loss, rather than the last optimization iterate.
        if best_continuous is not None:
            self.continuous.load_state_dict(best_continuous)
            self.surrogate.load_state_dict(best_surrogate)
        return self.history

    def save_history(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([rec.__dict__ for rec in self.history], f, indent=2)


__all__ = ["DiffPrepTrainer", "EpochRecord"]
