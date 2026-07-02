"""RL training stub for DeepPrep.

The default DeepPrep workflow only runs LLM **inference** (API or local), so
no training happens during the standard ``run_deepprep`` execution. This
module deliberately keeps the optional RL training surface minimal:

* :class:`RLTrainer` — collects trajectories by running a :class:`TreeAgent`
  on a list of tasks. When ``use_downstream_reward`` is ``True`` (default)
  the reward is the best small-data downstream metric (e.g. AUC) seen
  across the agent's ``<solution>`` attempts; failed runs receive ``0.0``.
  Set ``use_downstream_reward=False`` to recover the legacy binary reward
  that does not depend on downstream-model training.
* ``train()`` raises :class:`NotImplementedError` by default. Users who
  want to perform local SFT / PPO / GRPO updates should subclass and
  override ``train`` while sharing the local model with ``LLMClient``
  via ``LLMClient.attach_local_model``.
* :func:`cold_start_sft_skeleton` — placeholder to illustrate how the
  cold-start SFT phase would consume the same trajectories.

These hooks intentionally do **not** import ``trl`` / ``vllm`` so that the
default API-only path stays dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from baselines.common.pipeline import DataContext, Pipeline

from .evaluator import DeepPrepEvaluator
from .llm_client import LLMClient
from .sandbox import Sandbox
from .tree_agent import AgentRunResult, TreeAgent


@dataclass
class TaskSpec:
    """Description of a single training task fed to the RL trainer."""

    task_dir: str
    data_name: str
    data_dir: Optional[str] = None
    small_n: int = 0
    seed: int = 42
    ctx: Optional[DataContext] = None  # if None, inferred from sandbox


@dataclass
class Trajectory:
    """A single roll-out produced by ``TreeAgent.run``."""

    data_name: str
    success: bool
    reward: float
    n_turns: int
    n_errors: int
    transcript: list[dict] = field(default_factory=list)
    tree: dict = field(default_factory=dict)
    pipeline_ops: list[str] = field(default_factory=list)
    solution_attempts: list[dict] = field(default_factory=list)
    downstream_metrics: dict = field(default_factory=dict)


class RLTrainer:
    """Minimal RL training scaffold (NOT enabled by default).

    The class exposes two entry points:

    * :meth:`collect_trajectory` — run a single :class:`TreeAgent` rollout
      under the user-provided ``task`` and produce a :class:`Trajectory`.
      When ``use_downstream_reward`` is ``True`` (default) the reward is
      the best small-data downstream metric (e.g. AUC) across all
      ``<solution>`` attempts. When ``False`` the reward is binary
      (``1.0`` iff ``run.success``).
    * :meth:`train` — accepts a list of trajectories and is expected to
      perform a gradient update. The default implementation raises
      :class:`NotImplementedError`; subclass it to plug in SFT / PPO / GRPO.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_explore_turn: int = 5,
        max_chain_len: int = 6,
        max_depth: int = 8,
        max_err_cnt: int = 5,
        verbose: bool = False,
        use_downstream_reward: bool = True,
        downstream_eval_n: int = 3000,
        max_solution_attempts: int = 3,
    ) -> None:
        self.llm = llm
        self.max_explore_turn = int(max_explore_turn)
        self.max_chain_len = int(max_chain_len)
        self.max_depth = int(max_depth)
        self.max_err_cnt = int(max_err_cnt)
        self.verbose = bool(verbose)
        self.use_downstream_reward = bool(use_downstream_reward)
        self.downstream_eval_n = int(downstream_eval_n) if downstream_eval_n else 0
        self.max_solution_attempts = int(max_solution_attempts)

    # ------------------------------------------------------------------
    def collect_trajectory(self, task: TaskSpec) -> Trajectory:
        sandbox = Sandbox(
            task_dir=task.task_dir,
            data_name=task.data_name,
            data_dir=task.data_dir,
            small_n=task.small_n,
            seed=task.seed,
        )
        agent_evaluator: Optional[DeepPrepEvaluator] = None
        agent_eval_fn = None
        if self.use_downstream_reward:
            agent_evaluator = DeepPrepEvaluator(
                task_dir=task.task_dir,
                data_name=task.data_name,
                data_dir=task.data_dir,
                small_n=self.downstream_eval_n,
                seed=task.seed,
                verbose=False,
            )

            def _agent_eval_fn(pipeline: Pipeline):
                return agent_evaluator.evaluate_for_agent(pipeline)

            agent_eval_fn = _agent_eval_fn
        try:
            ctx = task.ctx
            if ctx is None:
                ctx = self._infer_ctx(sandbox, task.data_name)
            agent = TreeAgent(
                llm=self.llm,
                sandbox=sandbox,
                ctx=ctx,
                max_explore_turn=self.max_explore_turn,
                max_chain_len=self.max_chain_len,
                max_depth=self.max_depth,
                max_err_cnt=self.max_err_cnt,
                verbose=self.verbose,
                seed=task.seed,
                downstream_evaluator=agent_eval_fn,
                max_solution_attempts=self.max_solution_attempts,
            )
            run: AgentRunResult = agent.run()
        finally:
            sandbox.cleanup()

        # --- Compute reward ---
        downstream_metrics: dict = {}
        if self.use_downstream_reward:
            fits = [
                a["fitness"] for a in run.solution_attempts
                if isinstance(a.get("fitness"), float)
            ]
            if fits:
                best_idx = max(
                    range(len(run.solution_attempts)),
                    key=lambda i: (
                        run.solution_attempts[i]["fitness"]
                        if isinstance(run.solution_attempts[i].get("fitness"), float)
                        else float("-inf")
                    ),
                )
                reward = float(max(fits))
                downstream_metrics = dict(
                    run.solution_attempts[best_idx].get("metrics") or {}
                )
            else:
                reward = 0.0
        else:
            reward = 1.0 if run.success else 0.0

        return Trajectory(
            data_name=task.data_name,
            success=run.success,
            reward=reward,
            n_turns=run.n_turns,
            n_errors=run.n_errors,
            transcript=run.transcript,
            tree=run.tree,
            pipeline_ops=[s.op for s in run.pipeline.steps],
            solution_attempts=run.solution_attempts,
            downstream_metrics=downstream_metrics,
        )

    # ------------------------------------------------------------------
    def train(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        """Default implementation: NOT IMPLEMENTED.

        Subclass and override this method to perform a gradient update on a
        local model (which can be attached via
        ``LLMClient.attach_local_model``). The default refuses to train so
        that the standard DeepPrep verification path stays inference-only.
        """
        raise NotImplementedError(
            "Local RL training requires a fine-tunable model. "
            "Use backend='local' on LLMClient and subclass RLTrainer to "
            "implement your update rule (SFT / PPO / GRPO / etc.)."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_ctx(sandbox: Sandbox, data_name: str) -> DataContext:
        """Lazy import to avoid a hard dep on ``baselines.SAGA.saga`` at
        module import time."""
        from baselines.common.context import _infer_rec_context, _infer_tabular_context
        # Force an initial reset so ``sandbox.data`` is populated.
        if sandbox.data is None:
            sandbox.reset()
        if sandbox.task_type == "rec":
            return _infer_rec_context(data_name, {}, sandbox.data)
        return _infer_tabular_context(data_name, {}, sandbox.data)


# ---------------------------------------------------------------------------
def cold_start_sft_skeleton(
    trajectories: list[Trajectory],
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Placeholder for the Cold-Start SFT phase from the DeepPrep paper.

    A real implementation would:
      1. Filter ``trajectories`` to those with ``success=True``.
      2. Re-format each transcript into ``(prompt, response)`` pairs.
      3. Run a few SFT epochs on the local backbone before switching to RL.

    To keep the default DeepPrep install dependency-free we only return the
    aggregated stats here; subclass / replace this function to actually
    fine-tune.
    """
    n_total = len(trajectories)
    n_success = sum(1 for t in trajectories if t.success)
    return {
        "n_trajectories": n_total,
        "n_successful": n_success,
        "success_rate": (n_success / n_total) if n_total else 0.0,
        "output_dir": output_dir,
        "note": (
            "cold_start_sft_skeleton is a placeholder; subclass to implement "
            "an actual SFT update."
        ),
    }


__all__ = [
    "RLTrainer",
    "TaskSpec",
    "Trajectory",
    "cold_start_sft_skeleton",
]
