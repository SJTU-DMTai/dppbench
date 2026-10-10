"""Monte Carlo Tree Search core for BAT.

Faithfully reproduces ``ZJU-DAILY/BAT/src/mcts/mcts.py``:
  * Select  -- UCB1 with ``c * sqrt(ln(N_parent)/N_child)`` and
               unexplored children (N == 0) chosen first.
  * Expand  -- enumerate ``get_valid_action_space_for_node`` and create
               children via each action's ``create_children_nodes``.
  * Simulate -- random rollout (expand + uniform pick) until END.
  * Backpropagate -- single reward at END, propagated up.
  * Early stop when ``early_stop_n_paths`` END nodes attain reward exactly 1.
"""
from __future__ import annotations

import math
import random
import time
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from baselines.common.data_ctx import DataContext
from baselines.common.budget import BudgetExceededError, get_active_budget

from .action import (
    ActionContext,
    BaseAction,
    _pipeline_key,
    evaluate_downstream_feedback,
    get_valid_action_space_for_node,
)
from .node import MCTSNode, MCTSNodeType
from .reward import BATReward
from .llm_client import BATSearchTimeLimit

if TYPE_CHECKING:
    from baselines.DeepPrep.llm_client import LLMClient
    from .sandbox import Sandbox


@dataclass
class MCTSResult:
    root: MCTSNode
    end_nodes: list[MCTSNode] = field(default_factory=list)
    best_node: Optional[MCTSNode] = None
    selected_nodes: list[MCTSNode] = field(default_factory=list)
    best_paths: list[list[str]] = field(default_factory=list)
    n_rollouts: int = 0


class MCTSSolver:
    """BAT MCTS driver."""

    def __init__(
        self,
        llm: "LLMClient",
        ctx: DataContext,
        sandbox: "Sandbox",
        reward_model: BATReward,
        max_rollout_steps: int = 10,
        max_depth: int = 5,
        exploration_constant: float = 1.0,
        max_chain_len: int = 6,
        n_action_samples: int = 1,
        early_stop_n_paths: int = 2,
        early_stop_eps: float = 1e-3,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.ctx = ctx
        self.sandbox = sandbox
        self.reward_model = reward_model
        self.max_rollout_steps = int(max_rollout_steps)
        self.max_depth = int(max_depth)
        self.exploration_constant = float(exploration_constant)
        self.max_chain_len = int(max_chain_len)
        self.n_action_samples = int(n_action_samples)
        self.early_stop_n_paths = int(early_stop_n_paths)
        self.early_stop_eps = float(early_stop_eps)
        self.rng = random.Random(seed)
        self.verbose = bool(verbose)
        self.previous_pipeline_texts: list[str] = []

    # ------------------------------------------------------------------
    def _log(self, *a, **kw) -> None:
        if self.verbose:
            print("[BAT.MCTS]", *a, **kw)

    # ------------------------------------------------------------------
    def _action_ctx(self) -> ActionContext:
        return ActionContext(
            llm=self.llm,
            ctx=self.ctx,
            sandbox=self.sandbox,
            rng=self.rng,
            max_chain_len=self.max_chain_len,
            n_action_samples=self.n_action_samples,
            verbose=self.verbose,
            previous_pipeline_texts=self.previous_pipeline_texts,
            downstream_evaluator=getattr(self.reward_model, 'downstream_evaluator', None),
        )

    # ------------------------------------------------------------------
    def _ucb_score(self, child: MCTSNode, parent_N: int) -> float:
        if child.N == 0:
            return float("inf")
        exploit = child.Q / max(child.N, 1)
        explore = self.exploration_constant * math.sqrt(
            math.log(parent_N) / child.N
        )
        return exploit + explore

    def _restore_node_state(self, node: MCTSNode) -> bool:
        """Replay the sandbox to the data state represented by ``node``."""
        initial = getattr(self.sandbox, "_initial_snapshot", None)
        if initial is None:
            try:
                self.sandbox.reset()
                initial = getattr(self.sandbox, "_initial_snapshot", None)
            except Exception as e:
                self._log(f"failed to reset sandbox: {e}")
                return False
        try:
            self.sandbox.restore(initial)
            steps = list(node.final_pipeline_steps or node.latest_pipeline_steps())
            if steps:
                res = self.sandbox.execute_chain(steps)
                if not res.success:
                    node.exec_error = res.error or "node replay failed"
                    self._log(f"node replay failed: {node.exec_error}")
                    if node.node_type in (
                        MCTSNodeType.TRANSFORMATION,
                        MCTSNodeType.REVISED_TRANSFORMATION,
                    ):
                        # EPO needs failed transformations to remain expandable
                        # so TransformationRevision can consume the error.
                        self.sandbox.restore(initial)
                        return True
                    return False
            return True
        except Exception as e:
            node.exec_error = f"{type(e).__name__}: {e}"
            self._log(f"node replay raised: {node.exec_error}")
            return False

    def select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and not node.is_terminal():
            unvisited = next((child for child in node.children if child.N == 0), None)
            if unvisited is not None:
                return unvisited
            node = max(
                node.children,
                key=lambda child: self._ucb_score(child, node.N),
            )
        return node

    # ------------------------------------------------------------------
    def expand(self, node: MCTSNode) -> list[MCTSNode]:
        if node.is_terminal():
            return []
        if node.children:
            raise AssertionError("BAT expands only childless nodes")
        if not self._restore_node_state(node):
            return []
        if node.node_type == MCTSNodeType.TRANSFORMATION:
            evaluate_downstream_feedback(node, self._action_ctx())
        legal_actions = get_valid_action_space_for_node(node)
        if not legal_actions:
            return []
        new_children: list[MCTSNode] = []
        for action_cls in legal_actions:
            action: BaseAction = action_cls()
            try:
                children = action.create_children_nodes(node, self._action_ctx())
            except BATSearchTimeLimit:
                if new_children:
                    return new_children
                raise
            except BudgetExceededError:
                raise
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:
                self._log(f"action {action_cls.__name__} failed: {e}")
                continue
            for c in children:
                if c.is_terminal():
                    value, breakdown = self.reward_model.score(c)
                    c.reward_value = value
                    c.reward_breakdown = breakdown
                node.children.append(c)
                new_children.append(c)
        self.rng.shuffle(node.children)
        return new_children

    # ------------------------------------------------------------------
    def simulate(self, node: MCTSNode) -> MCTSNode:
        """Random rollout from ``node`` until an END node is reached."""
        cur = node
        while not cur.is_terminal():
            children = self.expand(cur)
            if not children:
                break
            cur = self.rng.choice(children)
        return cur

    # ------------------------------------------------------------------
    def backpropagate(self, end_node: MCTSNode, reward: float) -> None:
        cur: Optional[MCTSNode] = end_node
        while cur is not None:
            cur.N += 1
            cur.Q += reward
            cur = cur.parent

    # ------------------------------------------------------------------
    def solve(self) -> MCTSResult:
        root = MCTSNode(node_type=MCTSNodeType.ROOT, parent=None, depth=0)
        perfect_nodes: list[MCTSNode] = []
        seen_pipelines: set[str] = set()
        n_completed_rollouts = 0
        previous_rollout_seconds = 0.0

        for it in range(self.max_rollout_steps):
            budget = get_active_budget()
            if budget and n_completed_rollouts and budget.max_seconds is not None:
                remaining = budget.max_seconds - budget.elapsed_seconds
                if remaining < previous_rollout_seconds:
                    self._log(f"stopping before rollout {it + 1}: {remaining:.1f}s remain")
                    break
            rollout_started_at = time.monotonic()
            self._log(f"--- rollout {it + 1}/{self.max_rollout_steps} ---")
            # 1. Selection
            try:
                leaf = self.select(root)
                if not leaf.is_terminal():
                    children = self.expand(leaf)
                    if children:
                        leaf = self.rng.choice(children)
                    else:
                        continue
                end_node = leaf if leaf.is_terminal() else self.simulate(leaf)
            except BATSearchTimeLimit as exc:
                if not any(n.reward_value is not None for n in self._find_end_nodes(root)):
                    raise
                self._log(f"stopping with evaluated candidates: {exc}")
                break
            if not end_node.is_terminal():
                self._log("simulation failed to reach END; skipping rollout")
                continue

            # 4. Reward + backprop
            if end_node.reward_value is None:
                reward, breakdown = self.reward_model.score(end_node)
                end_node.reward_value = reward
                end_node.reward_breakdown = breakdown
            else:
                reward, breakdown = end_node.reward_value, end_node.reward_breakdown
            self.backpropagate(end_node, reward)
            n_completed_rollouts += 1
            previous_rollout_seconds = time.monotonic() - rollout_started_at

            self._log(
                f"end reward={reward:.4f} breakdown={breakdown} "
                f"ops={[s.op for s in end_node.final_pipeline_steps]}"
            )

            pipeline_key = _pipeline_key(end_node.final_pipeline_steps)
            is_new_pipeline = pipeline_key not in seen_pipelines
            if is_new_pipeline:
                seen_pipelines.add(pipeline_key)
                chain = " --> ".join(
                    f"{step.op}(" + ", ".join(
                        f"{key}={value!r}" for key, value in step.params.items()
                    ) + ")" for step in end_node.final_pipeline_steps
                )
                self.previous_pipeline_texts.append(chain)
            if reward == 1.0 and is_new_pipeline:
                perfect_nodes.append(end_node)
                if len(perfect_nodes) >= self.early_stop_n_paths:
                    self._log(
                        f"early stop after {len(perfect_nodes)} perfect paths "
                        f"at rollout {it + 1}"
                    )
                    break

        end_nodes = self._find_end_nodes(root)
        selected_nodes = list(perfect_nodes[: self.early_stop_n_paths])
        selected_keys = {_pipeline_key(node.final_pipeline_steps) for node in selected_nodes}
        if len(selected_nodes) < self.early_stop_n_paths:
            ranked = sorted(
                [node for node in end_nodes if node.reward_value is not None],
                key=lambda node: node.Q / node.N if node.N else node.reward_value,
                reverse=True,
            )
            for node in ranked:
                key = _pipeline_key(node.final_pipeline_steps)
                if key in selected_keys:
                    continue
                selected_nodes.append(node)
                selected_keys.add(key)
                if len(selected_nodes) >= self.early_stop_n_paths:
                    break
        best_paths = [
            [node.label() for node in selected.path_to_root()]
            for selected in selected_nodes
        ]
        best_node = selected_nodes[0] if selected_nodes else None

        return MCTSResult(
            root=root,
            end_nodes=end_nodes,
            best_node=best_node,
            selected_nodes=selected_nodes,
            best_paths=best_paths,
            n_rollouts=n_completed_rollouts,
        )

    def _find_end_nodes(self, node: MCTSNode) -> list[MCTSNode]:
        if node.is_terminal():
            return [node]
        result: list[MCTSNode] = []
        for child in node.children:
            result.extend(self._find_end_nodes(child))
        return result


__all__ = ["MCTSSolver", "MCTSResult"]
