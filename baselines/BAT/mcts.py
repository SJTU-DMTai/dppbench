"""Monte Carlo Tree Search core for BAT.

Faithfully reproduces ``ZJU-DAILY/BAT/src/mcts/mcts.py``:
  * Select  -- UCB1 with ``c * sqrt(ln(N_parent)/N_child)`` and
               unexplored children (N == 0) chosen first.
  * Expand  -- enumerate ``get_valid_action_space_for_node`` and create
               children via each action's ``create_children_nodes``.
  * Simulate -- random rollout (expand + uniform pick) until END.
  * Backpropagate -- single reward at END, propagated up.
  * Early stop when ``early_stop_n_paths`` END nodes attain reward
    ``>= 1 - eps``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from baselines.SAGA.pipeline import DataContext

from .action import (
    ActionContext,
    BaseAction,
    EndAction,
    expected_target_columns,
    get_valid_action_space_for_node,
)
from .node import MCTSNode, MCTSNodeType
from .reward import BATReward

if TYPE_CHECKING:
    from baselines.DeepPrep.llm_client import LLMClient
    from .sandbox import Sandbox


@dataclass
class MCTSResult:
    root: MCTSNode
    end_nodes: list[MCTSNode] = field(default_factory=list)
    best_node: Optional[MCTSNode] = None
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
        self.early_stop_n_paths = int(early_stop_n_paths)
        self.early_stop_eps = float(early_stop_eps)
        self.rng = random.Random(seed)
        self.verbose = bool(verbose)

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
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    def _ucb_score(self, child: MCTSNode, parent_N: int) -> float:
        if child.N == 0:
            return float("inf")
        exploit = child.Q / max(child.N, 1)
        explore = self.exploration_constant * math.sqrt(
            math.log(max(parent_N, 1) + 1) / max(child.N, 1)
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
                    return False
            return True
        except Exception as e:
            node.exec_error = f"{type(e).__name__}: {e}"
            self._log(f"node replay raised: {node.exec_error}")
            return False

    def select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.children and not node.is_terminal():
            scores = [self._ucb_score(c, node.N) for c in node.children]
            best = max(range(len(node.children)), key=lambda i: scores[i])
            node = node.children[best]
        return node

    # ------------------------------------------------------------------
    def expand(self, node: MCTSNode) -> list[MCTSNode]:
        if node.is_terminal() or node.depth >= self.max_depth:
            return []
        if not self._restore_node_state(node):
            return []
        legal_actions = get_valid_action_space_for_node(node)
        if not legal_actions:
            return []
        new_children: list[MCTSNode] = []
        for action_cls in legal_actions:
            action: BaseAction = action_cls()
            try:
                children = action.create_children_nodes(node, self._action_ctx())
            except Exception as e:
                self._log(f"action {action_cls.__name__} failed: {e}")
                continue
            for c in children:
                node.children.append(c)
                new_children.append(c)
        return new_children

    # ------------------------------------------------------------------
    def simulate(self, node: MCTSNode) -> MCTSNode:
        """Random rollout from ``node`` until an END node is reached."""
        cur = node
        while not cur.is_terminal() and cur.depth < self.max_depth:
            children = list(cur.children) if cur.children else self.expand(cur)
            if not children:
                # Force-finalize if nothing legal exists.
                if not self._restore_node_state(cur):
                    break
                end_children = EndAction().create_children_nodes(
                    cur, self._action_ctx()
                )
                if not end_children:
                    break
                cur.children.extend(end_children)
                cur = end_children[0]
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
        root.N = 1  # avoid log(0) in UCB
        end_nodes: list[MCTSNode] = []
        best_paths: list[list[str]] = []

        for it in range(self.max_rollout_steps):
            self._log(f"--- rollout {it + 1}/{self.max_rollout_steps} ---")
            # 1. Selection
            leaf = self.select(root)

            # 2. Expansion (skip if leaf is terminal)
            if not leaf.is_terminal() and leaf.depth < self.max_depth:
                children = self.expand(leaf)
                if children:
                    leaf = self.rng.choice(children)

            # 3. Simulation
            end_node = leaf if leaf.is_terminal() else self.simulate(leaf)
            if not end_node.is_terminal():
                # Force terminal even if max_depth tripped before END.
                if not self._restore_node_state(end_node):
                    continue
                forced = EndAction().create_children_nodes(
                    end_node, self._action_ctx()
                )
                if forced:
                    end_node.children.extend(forced)
                    end_node = forced[0]

            if not end_node.is_terminal():
                self._log("simulation failed to reach END; skipping rollout")
                continue

            # 4. Reward + backprop
            reward, breakdown = self.reward_model.score(end_node)
            self.backpropagate(end_node, reward)
            end_nodes.append(end_node)

            self._log(
                f"end reward={reward:.4f} breakdown={breakdown} "
                f"ops={[s.op for s in end_node.final_pipeline_steps]}"
            )

            if reward >= 1.0 - self.early_stop_eps:
                best_paths.append([n.label() for n in end_node.path_to_root()])
                if len(best_paths) >= self.early_stop_n_paths:
                    self._log(
                        f"early stop after {len(best_paths)} perfect paths "
                        f"at rollout {it + 1}"
                    )
                    break

        # Pick best by reward (tie-break by downstream_fitness)
        best_node: Optional[MCTSNode] = None
        for node in end_nodes:
            if best_node is None:
                best_node = node
                continue
            best_r = best_node.reward_value or 0.0
            node_r = node.reward_value or 0.0
            if node_r > best_r:
                best_node = node
            elif abs(node_r - best_r) < 1e-9:
                best_d = best_node.downstream_fitness or -1.0
                node_d = node.downstream_fitness or -1.0
                if node_d > best_d:
                    best_node = node

        return MCTSResult(
            root=root,
            end_nodes=end_nodes,
            best_node=best_node,
            best_paths=best_paths,
            n_rollouts=len(end_nodes),
        )


__all__ = ["MCTSSolver", "MCTSResult"]
