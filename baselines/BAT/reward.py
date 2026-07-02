"""BAT reward model.

The original BAT reward only looked at *column similarity* between the
synthesized table and the (target-instance-free) target schema. dppbench
additionally requires a downstream ML signal, so this reward fuses three
sources:

  reward = alpha * column_similarity
         + beta  * downstream_metric  (LightGBM AUC / DIN AUC)
         + gamma * llm_judge          (optional, in {0, 0.5, 1})

with ``alpha + beta + gamma`` typically equal to 1.0. Setting
``use_downstream=False`` recovers the original BAT semantics exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from baselines.common.pipeline import DataContext, Pipeline

from . import prompts
from .action import (
    _column_similarity,
    _sandbox_columns,
    expected_target_columns,
)
from .node import MCTSNode

if TYPE_CHECKING:
    from baselines.DeepPrep.llm_client import LLMClient
    from .evaluator import BATEvaluator
    from .sandbox import Sandbox


_REWARD_TAG_RE = re.compile(r"<reward>(.*?)</reward>", re.DOTALL)


def _parse_llm_reward(text: str) -> Optional[float]:
    m = _REWARD_TAG_RE.search(text or "")
    raw = (m.group(1) if m else text or "").strip()
    try:
        v = float(raw)
    except Exception:
        return None
    if v >= 1.0:
        return 1.0
    if v >= 0.5:
        return 0.5
    return 0.0


@dataclass
class BATRewardConfig:
    alpha: float = 0.4
    beta: float = 0.5
    gamma: float = 0.1
    use_downstream: bool = True
    use_llm_judge: bool = False
    columns_match_threshold: float = 0.99


class BATReward:
    """Compute the fused BAT reward for a terminal MCTS node.

    The reward function is invoked exactly once per END node by the
    MCTS solver during backpropagation. Side effects: it materialises
    ``end_node.column_similarity / downstream_fitness / reward_value /
    reward_breakdown`` for downstream serialisation.
    """

    def __init__(
        self,
        ctx: DataContext,
        sandbox: "Sandbox",
        downstream_evaluator: Optional["BATEvaluator"] = None,
        llm: Optional["LLMClient"] = None,
        config: Optional[BATRewardConfig] = None,
    ) -> None:
        self.ctx = ctx
        self.sandbox = sandbox
        self.downstream_evaluator = downstream_evaluator
        self.llm = llm
        self.config = config or BATRewardConfig()

    # ------------------------------------------------------------------
    def score(self, end_node: MCTSNode) -> tuple[float, dict]:
        cfg = self.config
        breakdown: dict = {}

        steps = list(end_node.final_pipeline_steps
                     or end_node.latest_pipeline_steps())
        pipeline = Pipeline(steps=steps)

        # ---- 1. Execute the pipeline in the sandbox to compute column sim
        column_similarity = 0.0
        columns_match = False
        exec_error: Optional[str] = None
        if not steps:
            exec_error = "empty pipeline"
        else:
            self.sandbox.restore(self.sandbox._initial_snapshot)  # type: ignore[arg-type, attr-defined]
            exec_res = self.sandbox.execute_chain(steps)
            if not exec_res.success:
                exec_error = exec_res.error
            else:
                expected = expected_target_columns(self.ctx)
                actual = _sandbox_columns(self.sandbox)
                column_similarity = _column_similarity(actual, expected)
                columns_match = column_similarity >= cfg.columns_match_threshold

        end_node.exec_error = exec_error
        end_node.column_similarity = column_similarity
        end_node.columns_match = columns_match
        breakdown["column_similarity"] = column_similarity
        breakdown["columns_match"] = columns_match
        breakdown["exec_error"] = exec_error

        # ---- 2. Downstream training (optional) -------------------------
        downstream_score = 0.0
        downstream_fitness: Optional[float] = None
        downstream_metrics: dict = {}
        downstream_error: Optional[str] = None
        if cfg.use_downstream and self.downstream_evaluator is not None and not exec_error:
            (
                downstream_fitness,
                downstream_metrics,
                downstream_error,
            ) = self.downstream_evaluator.evaluate_for_agent(pipeline)
            if isinstance(downstream_fitness, float):
                downstream_score = max(0.0, min(1.0, downstream_fitness))
        end_node.downstream_fitness = downstream_fitness
        end_node.downstream_metrics = downstream_metrics or {}
        breakdown["downstream_fitness"] = downstream_fitness
        breakdown["downstream_metrics"] = downstream_metrics
        breakdown["downstream_error"] = downstream_error

        # ---- 3. LLM judge (optional) -----------------------------------
        llm_score = 0.0
        if cfg.use_llm_judge and self.llm is not None:
            obs_text = ""
            try:
                obs_text = self.sandbox._observe().text  # type: ignore[attr-defined]
            except Exception:
                obs_text = ""
            judge_prompt = prompts.render_reward_judge(
                task_type=self.ctx.task_type,
                data_name=self.ctx.data_name,
                obs_after_exec=obs_text,
                pipeline_ops=[s.op for s in steps],
                columns_match=columns_match,
                column_similarity=column_similarity,
                downstream_fitness=downstream_fitness,
                exec_error=exec_error,
            )
            try:
                resp = self.llm.chat(
                    [
                        {"role": "system", "content": prompts.SYSTEM_BAT},
                        {"role": "user", "content": judge_prompt},
                    ]
                )
                parsed = _parse_llm_reward(resp)
                if parsed is not None:
                    llm_score = parsed
            except Exception:
                llm_score = 0.0
        breakdown["llm_judge"] = llm_score

        # ---- 4. Fuse ----------------------------------------------------
        weighted_alpha = cfg.alpha
        weighted_beta = cfg.beta if cfg.use_downstream else 0.0
        weighted_gamma = cfg.gamma if cfg.use_llm_judge else 0.0
        denom = weighted_alpha + weighted_beta + weighted_gamma
        if denom <= 0:
            reward = 0.0
        else:
            reward = (
                weighted_alpha * column_similarity
                + weighted_beta * downstream_score
                + weighted_gamma * llm_score
            ) / denom

        # If pipeline did not execute, force a 0 reward so the search
        # avoids it.
        if exec_error:
            reward = 0.0

        end_node.reward_value = reward
        end_node.reward_breakdown = breakdown
        initial = getattr(self.sandbox, "_initial_snapshot", None)
        if initial is not None:
            try:
                self.sandbox.restore(initial)
            except Exception:
                pass
        return reward, breakdown


__all__ = ["BATReward", "BATRewardConfig"]
