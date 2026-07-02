"""Search-tree data structures for the DeepPrep tree agent.

A node corresponds to a sandbox state reached by executing a particular
prefix of operators. Children represent successful expansions; backtracking
moves the cursor up the tree.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from baselines.common.pipeline import PipelineStep


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
    """Indexed collection of ``SearchNode`` objects.

    The tree owns the ``cursor`` (the node the agent is currently expanding
    from). ``add_child`` advances the cursor; ``backtrack`` moves it back.
    """

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
        """Move the cursor to the parent of the current node. Returns the
        new cursor node, or ``None`` if already at the root.
        """
        if self.cursor_id is None:
            return None
        cur = self.nodes[self.cursor_id]
        if cur.parent_id is None:
            return None
        self.cursor_id = cur.parent_id
        return self.nodes[self.cursor_id]

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
                    "is_terminal": n.is_terminal,
                    "error": n.error,
                    "children_ids": list(n.children_ids),
                }
                for n in self.nodes.values()
            ],
        }


__all__ = ["SearchNode", "SearchTree"]
