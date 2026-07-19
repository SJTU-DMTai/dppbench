"""Search-tree data structures for the DeepPrep tree agent.

A node corresponds to a materialized sandbox state reached by executing one
operator. Children represent successful single-operator expansions; paths from
root to a node represent complete pipeline prefixes.
"""
from __future__ import annotations

import uuid
import json
from dataclasses import dataclass, field
from typing import Optional

from baselines.common.pipeline import PipelineStep


def canonical_step(step: PipelineStep) -> str:
    """Stable operator-instance text used in DeepPrep parent paths."""
    params = step.params or {}
    if not params:
        return step.op
    rendered = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)}"
        for key, value in sorted(params.items())
    )
    return f"{step.op}({rendered})"


def canonical_path(steps: list[PipelineStep]) -> str:
    return " --> ".join(canonical_step(step) for step in steps) or "root"


@dataclass
class SearchNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    accumulated_steps: list[PipelineStep] = field(default_factory=list)
    snapshot: Optional[bytes] = None       # sandbox.snapshot() bytes
    obs_text: str = ""                     # rendered observation
    is_terminal: bool = False
    error: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)


class SearchTree:
    """Indexed collection of materialized DeepPrep reasoning states."""

    def __init__(self) -> None:
        self.nodes: dict[str, SearchNode] = {}
        self.root_id: Optional[str] = None
        self.cursor_id: Optional[str] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:8]

    def add_root(self, snapshot: bytes, obs_text: str) -> SearchNode:
        node = SearchNode(
            node_id=self._new_id(),
            parent_id=None,
            depth=0,
            accumulated_steps=[],
            snapshot=snapshot,
            obs_text=obs_text,
        )
        self.nodes[node.node_id] = node
        self.root_id = node.node_id
        self.cursor_id = node.node_id
        return node

    def add_child(
        self,
        new_steps: list[PipelineStep],
        snapshot: bytes,
        obs_text: str,
    ) -> SearchNode:
        if self.cursor_id is None:
            raise RuntimeError("SearchTree.add_child() before add_root()")
        if len(new_steps) != 1:
            raise ValueError("DeepPrep tree edges must contain exactly one operator")
        parent = self.nodes[self.cursor_id]
        child = SearchNode(
            node_id=self._new_id(),
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            accumulated_steps=parent.accumulated_steps + list(new_steps),
            snapshot=snapshot,
            obs_text=obs_text,
        )
        self.nodes[child.node_id] = child
        parent.children_ids.append(child.node_id)
        self.cursor_id = child.node_id
        return child

    def backtrack(self) -> Optional[SearchNode]:
        """Move the cursor to the parent of the current node."""
        if self.cursor_id is None:
            return None
        cur = self.nodes[self.cursor_id]
        if cur.parent_id is None:
            return None
        self.cursor_id = cur.parent_id
        return self.nodes[self.cursor_id]

    def set_cursor(self, node_id: str) -> SearchNode:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown DeepPrep tree node: {node_id}")
        self.cursor_id = node_id
        return self.nodes[node_id]

    def find_by_steps(self, steps: list[PipelineStep]) -> Optional[SearchNode]:
        """Find a node by its complete, parameterized root-to-node path."""
        expected = canonical_path(steps)
        for node in self.nodes.values():
            if canonical_path(node.accumulated_steps) == expected:
                return node
        return None

    def find_by_path(self, path: str) -> Optional[SearchNode]:
        normalized = path.strip()
        if normalized.lower() in ("", "root", "<root>"):
            return self.nodes.get(self.root_id) if self.root_id else None
        for node in self.nodes.values():
            if canonical_path(node.accumulated_steps) == normalized:
                return node
        return None

    def mark_cursor_error(self, error: str | None) -> None:
        if self.cursor_id is not None:
            self.nodes[self.cursor_id].error = error

    # ------------------------------------------------------------------
    @property
    def cursor(self) -> SearchNode:
        if self.cursor_id is None:
            raise RuntimeError("SearchTree has no cursor (uninitialised)")
        return self.nodes[self.cursor_id]

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "cursor_id": self.cursor_id,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "accumulated_ops": [s.op for s in n.accumulated_steps],
                    "path": canonical_path(n.accumulated_steps),
                    "accumulated_steps": [s.to_dict() for s in n.accumulated_steps],
                    "is_terminal": n.is_terminal,
                    "error": n.error,
                    "children_ids": list(n.children_ids),
                }
                for n in self.nodes.values()
            ],
        }


__all__ = ["SearchNode", "SearchTree", "canonical_step", "canonical_path"]
