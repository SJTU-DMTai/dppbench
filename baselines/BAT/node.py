"""MCTS node definitions for BAT.

A faithful port of ``ZJU-DAILY/BAT/src/mcts/node.py`` adapted to dppbench's
operator-pipeline representation. Five node types implement BAT's DPAS
(Data Preparation Action Sandbox); the legality table
:func:`get_valid_action_space_for_node` mirrors the original BAT logic
modulo "no repeated action types on the same path".
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from baselines.common.pipeline import PipelineStep

if TYPE_CHECKING:
    from .action import BaseAction


class MCTSNodeType(str, enum.Enum):
    ROOT = "ROOT"
    SCHEMA_MATCH = "SCHEMA_MATCH"
    IDENTIFY_COLUMN_FUNCTIONS = "IDENTIFY_COLUMN_FUNCTIONS"
    TRANSFORMATION = "TRANSFORMATION"
    REVISED_TRANSFORMATION = "REVISED_TRANSFORMATION"
    END = "END"


@dataclass
class MCTSNode:
    """A node in the BAT search tree.

    Pipeline state is represented as ``list[PipelineStep]`` so dppbench's
    YAML executor can run it directly. Every node also carries the
    sandbox-side metrics needed by :class:`BATReward` (column similarity,
    downstream fitness, exec error).
    """

    node_type: MCTSNodeType
    parent: Optional["MCTSNode"] = None
    parent_action: Optional["BaseAction"] = None
    depth: int = 0

    # ---- DPAS payload ----
    schema_match: Optional[dict] = None
    column_functions: Optional[str] = None
    pipeline_steps: list[PipelineStep] = field(default_factory=list)
    revised_pipeline_steps: list[PipelineStep] = field(default_factory=list)

    # ---- EPO outputs (filled when node_type == END) ----
    final_pipeline_steps: list[PipelineStep] = field(default_factory=list)
    columns_match: Optional[bool] = None
    column_similarity: Optional[float] = None
    downstream_fitness: Optional[float] = None
    downstream_metrics: dict = field(default_factory=dict)
    exec_error: Optional[str] = None
    reward_value: Optional[float] = None
    reward_breakdown: dict = field(default_factory=dict)

    # ---- MCTS bookkeeping ----
    Q: float = 0.0
    N: int = 0
    children: list["MCTSNode"] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.node_type == MCTSNodeType.END

    def path_node_types(self) -> list[MCTSNodeType]:
        nodes: list[MCTSNodeType] = []
        cur: Optional["MCTSNode"] = self
        while cur is not None:
            nodes.append(cur.node_type)
            cur = cur.parent
        nodes.reverse()
        return nodes

    def path_to_root(self) -> list["MCTSNode"]:
        path: list["MCTSNode"] = []
        cur: Optional["MCTSNode"] = self
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path.reverse()
        return path

    def latest_pipeline_steps(self) -> list[PipelineStep]:
        """Walk from root to ``self``, returning the last non-empty
        pipeline emitted by any ancestor (revision wins over original).
        """
        latest: list[PipelineStep] = []
        for node in self.path_to_root():
            if node.revised_pipeline_steps:
                latest = list(node.revised_pipeline_steps)
            elif node.pipeline_steps:
                latest = list(node.pipeline_steps)
        return latest

    def label(self) -> str:
        return self.node_type.value

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type.value,
            "depth": self.depth,
            "Q": self.Q,
            "N": self.N,
            "schema_match": self.schema_match,
            "column_functions": self.column_functions,
            "pipeline_ops": [s.op for s in self.pipeline_steps],
            "revised_pipeline_ops": [s.op for s in self.revised_pipeline_steps],
            "final_pipeline_ops": [s.op for s in self.final_pipeline_steps],
            "columns_match": self.columns_match,
            "column_similarity": self.column_similarity,
            "downstream_fitness": self.downstream_fitness,
            "downstream_metrics": self.downstream_metrics,
            "exec_error": self.exec_error,
            "reward_value": self.reward_value,
            "reward_breakdown": self.reward_breakdown,
            "children": [c.to_dict() for c in self.children],
        }


__all__ = ["MCTSNode", "MCTSNodeType"]
