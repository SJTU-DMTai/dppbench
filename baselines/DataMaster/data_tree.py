"""DataTree data structures for the DataMaster baseline.

Implements the multi-branch search tree from the DataMaster paper §3.3.
Each node v stores the *delta* operator chain it applied on top of its
parent and an *accumulated* pipeline (parent.acc + delta) so a sub-branch
inherits the full upstream context. UCB statistics ``visits`` and
``total_reward`` are kept on every node and updated by ``backpropagate``.

Red-Node / Data-Pool branch is intentionally absent — DataMaster in this
repo runs a black-node-only tree.
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from baselines.SAGA.pipeline import PipelineStep


# ---------------------------------------------------------------------------
# Per-node record  (paper: r_v = (black, D_v, y_v, phi_v))
# ---------------------------------------------------------------------------
@dataclass
class NodeRecord:
    """Single node in the DataTree.

    Notes
    -----
    * ``accumulated_steps`` is the FULL pipeline produced by this node
      (parent.accumulated + delta). ``delta_steps`` is just the increment so
      the ancestry is reconstructable.
    * ``fitness`` corresponds to the paper's ``y_v``; ``metrics`` and
      ``diagnostics`` jointly form ``phi_v``. ``error`` is ``None`` for a
      successfully evaluated node.
    * ``findings`` are short free-form strings the LLM writes via the
      Global Memory protocol (``<finding>...</finding>`` tags).
    * ``visits`` / ``total_reward`` implement UCB statistics. The reward
      semantics (raw fitness vs improvement) is decided by
      :class:`UCBScheduler` and applied in :func:`backpropagate`.
    """

    node_id: str
    parent_id: Optional[str]
    depth: int
    delta_steps: list[PipelineStep] = field(default_factory=list)
    accumulated_steps: list[PipelineStep] = field(default_factory=list)
    pipeline_ops: list[str] = field(default_factory=list)

    # Evaluation outcome (None for the un-evaluated root)
    fitness: Optional[float] = None
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)

    findings: list[str] = field(default_factory=list)

    # Tree topology / control
    children_ids: list[str] = field(default_factory=list)
    is_terminal: bool = False
    is_root: bool = False

    # UCB statistics
    visits: int = 0
    total_reward: float = 0.0

    created_at: float = field(default_factory=time.time)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "is_root": self.is_root,
            "is_terminal": self.is_terminal,
            "delta_ops": [s.op for s in self.delta_steps],
            "accumulated_ops": [s.op for s in self.accumulated_steps],
            "fitness": self.fitness,
            "metrics": self.metrics,
            "error": self.error,
            "diagnostics": self.diagnostics,
            "findings": list(self.findings),
            "children_ids": list(self.children_ids),
            "visits": self.visits,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# DataTree
# ---------------------------------------------------------------------------
class DataTree:
    """Multi-branch search tree used by DataMaster.

    The tree owns no Sandbox snapshots: parents are re-played by the agent
    via the SAGA Pipeline ``accumulated_steps`` to avoid bloating
    pickle-heavy state. Snapshot reuse is delegated to the Sandbox (see
    :mod:`baselines.DeepPrep.sandbox`).
    """

    def __init__(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}
        self.root_id: Optional[str] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:10]

    # ------------------------------------------------------------------
    def add_root(
        self,
        initial_steps: Optional[list[PipelineStep]] = None,
    ) -> NodeRecord:
        """Create the root node v_0 (paper: initial data state).

        The root is NOT evaluated and never appears as ``best_node``; it
        only acts as the parent of the first wave of black children.
        """
        if self.root_id is not None:
            raise RuntimeError("DataTree already has a root.")
        steps = list(initial_steps or [])
        node = NodeRecord(
            node_id=self._new_id(),
            parent_id=None,
            depth=0,
            delta_steps=copy.deepcopy(steps),
            accumulated_steps=copy.deepcopy(steps),
            pipeline_ops=[s.op for s in steps],
            is_root=True,
        )
        self.nodes[node.node_id] = node
        self.root_id = node.node_id
        return node

    # ------------------------------------------------------------------
    def add_black_child(
        self,
        parent_id: str,
        delta_steps: list[PipelineStep],
        *,
        fitness: Optional[float],
        metrics: Optional[dict] = None,
        error: Optional[str] = None,
        diagnostics: Optional[dict] = None,
        findings: Optional[list[str]] = None,
    ) -> NodeRecord:
        """Append a black child to ``parent_id``.

        ``accumulated_steps`` is computed as parent.acc + delta so the
        downstream evaluator can simply consume the resulting Pipeline.
        """
        if parent_id not in self.nodes:
            raise KeyError(f"Unknown parent_id={parent_id!r}")
        parent = self.nodes[parent_id]
        delta = copy.deepcopy(list(delta_steps))
        accumulated = copy.deepcopy(parent.accumulated_steps) + delta
        node = NodeRecord(
            node_id=self._new_id(),
            parent_id=parent_id,
            depth=parent.depth + 1,
            delta_steps=delta,
            accumulated_steps=accumulated,
            pipeline_ops=[s.op for s in accumulated],
            fitness=fitness,
            metrics=dict(metrics or {}),
            error=error,
            diagnostics=dict(diagnostics or {}),
            findings=list(findings or []),
        )
        self.nodes[node.node_id] = node
        parent.children_ids.append(node.node_id)
        return node

    # ------------------------------------------------------------------
    def parent(self, node_id: str) -> Optional[NodeRecord]:
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.nodes.get(node.parent_id)

    def siblings(self, node_id: str) -> list[NodeRecord]:
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return []
        parent = self.nodes[node.parent_id]
        return [
            self.nodes[cid]
            for cid in parent.children_ids
            if cid != node_id and cid in self.nodes
        ]

    def frontier(self) -> list[NodeRecord]:
        """All non-terminal nodes (the root is included until it has at
        least one successful child to anchor expansion).
        """
        return [n for n in self.nodes.values() if not n.is_terminal]

    def has_frontier(self) -> bool:
        return any(not n.is_terminal for n in self.nodes.values())

    def best_node(self) -> Optional[NodeRecord]:
        """Argmax over evaluated black nodes by ``fitness``.

        The root is excluded (it has ``fitness=None``) and so are nodes that
        failed to evaluate. Returns ``None`` when no successful black node
        exists.
        """
        best: Optional[NodeRecord] = None
        for node in self.nodes.values():
            if node.is_root:
                continue
            if node.fitness is None:
                continue
            if best is None or node.fitness > best.fitness:
                best = node
        return best

    def evaluated_nodes(self) -> list[NodeRecord]:
        return [n for n in self.nodes.values() if (not n.is_root) and n.fitness is not None]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "n_nodes": len(self.nodes),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }


# ---------------------------------------------------------------------------
# Reward backpropagation
# ---------------------------------------------------------------------------
def backpropagate(tree: DataTree, leaf_id: str, reward: float) -> None:
    """Walk from ``leaf_id`` up to the root, incrementing ``visits`` and
    accumulating ``reward`` on every ancestor (the leaf included).

    Implements the standard MCTS update used in DataMaster §3.5.
    """
    cur = tree.nodes.get(leaf_id)
    while cur is not None:
        cur.visits += 1
        cur.total_reward += float(reward)
        if cur.parent_id is None:
            break
        cur = tree.nodes.get(cur.parent_id)


__all__ = [
    "NodeRecord",
    "DataTree",
    "backpropagate",
]
