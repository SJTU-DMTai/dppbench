"""Global Memory for the DataMaster baseline.

Implements the paper's §3.4 cumulative memory: each node already owns its
record ``r_v = (black, D_v, y_v, phi_v)`` via :class:`NodeRecord`; this
module adds retrieval and prompt-friendly formatting.

Default retrieval window for node ``v`` is ``M_v = {Par(v)} ∪ Sib(v)``;
we additionally inject the global top-K best-evaluated nodes so successful
processing choices propagate across branches (paper: "transfer successful
processing choices via memory").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .data_tree import DataTree, NodeRecord


_TRUNC_FINDINGS = 3
_TRUNC_OPS = 12


def _format_metrics(m: Optional[dict]) -> str:
    if not m:
        return "{}"
    parts = []
    for k, v in m.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"


def _format_ops(ops: list[str], limit: int = _TRUNC_OPS) -> str:
    if not ops:
        return "(empty)"
    if len(ops) <= limit:
        return " -> ".join(ops)
    return " -> ".join(ops[:limit]) + f" ... (+{len(ops) - limit} more)"


def _node_brief(node: NodeRecord, role: str) -> list[str]:
    fit_str = f"{node.fitness:.4f}" if isinstance(node.fitness, float) else "n/a"
    lines = [
        f"### {role} (node={node.node_id}, depth={node.depth}, "
        f"visits={node.visits}, mean_R={node.mean_reward:.4f})",
        f"- ops: {_format_ops(node.pipeline_ops)}",
        f"- fitness: {fit_str}",
    ]
    if node.metrics:
        lines.append(f"- metrics: {_format_metrics(node.metrics)}")
    if node.error:
        err = node.error.splitlines()[0][:160]
        lines.append(f"- error: {err}")
    if node.findings:
        for f in node.findings[-_TRUNC_FINDINGS:]:
            lines.append(f"- finding: {f.strip()[:200]}")
    return lines


@dataclass
class GlobalMemory:
    """Memory wrapper around a :class:`DataTree`.

    Note that we do NOT copy data: ``retrieve`` returns live
    :class:`NodeRecord` references; mutation goes through
    :meth:`write_finding`.
    """

    tree: DataTree
    top_k_global: int = 3
    max_chars: int = 4000
    log: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    def retrieve(self, node_id: str) -> dict:
        node = self.tree.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown node_id={node_id!r}")
        parent = self.tree.parent(node_id)
        siblings = self.tree.siblings(node_id)
        # Sort siblings by recency (most recent first)
        siblings = sorted(siblings, key=lambda n: n.created_at, reverse=True)
        global_top = self._top_k_global(exclude={node_id, *(s.node_id for s in siblings)})
        return {"parent": parent, "siblings": siblings, "global_top_k": global_top}

    # ------------------------------------------------------------------
    def _top_k_global(self, exclude: Optional[set] = None) -> list[NodeRecord]:
        exclude = exclude or set()
        evaluated = [
            n for n in self.tree.evaluated_nodes()
            if n.node_id not in exclude
        ]
        evaluated.sort(key=lambda n: (n.fitness or float("-inf")), reverse=True)
        return evaluated[: self.top_k_global]

    # ------------------------------------------------------------------
    def format_context(self, retrieved: dict) -> str:
        """Render retrieved records as markdown for prompt injection.

        Truncated to ``self.max_chars`` so the operator catalog and the
        observation itself still fit in the LLM's context window.
        """
        lines: list[str] = []

        parent: Optional[NodeRecord] = retrieved.get("parent")
        siblings: list[NodeRecord] = list(retrieved.get("siblings") or [])
        global_top: list[NodeRecord] = list(retrieved.get("global_top_k") or [])

        if parent is not None and not parent.is_root:
            lines.append("## Memory: Parent Node")
            lines.extend(_node_brief(parent, "Parent"))
            lines.append("")
        elif parent is not None and parent.is_root:
            lines.append("## Memory: Parent Node")
            lines.append(
                "### Parent (root) — initial dataset, no operators applied yet."
            )
            lines.append("")

        if siblings:
            lines.append(f"## Memory: Sibling Nodes (n={len(siblings)})")
            for i, sib in enumerate(siblings[:5]):
                lines.extend(_node_brief(sib, f"Sibling[{i}]"))
            if len(siblings) > 5:
                lines.append(f"(+{len(siblings) - 5} more siblings hidden)")
            lines.append("")

        if global_top:
            lines.append(
                f"## Memory: Global Top-{len(global_top)} (cross-branch best)"
            )
            for i, n in enumerate(global_top):
                lines.extend(_node_brief(n, f"GlobalBest[{i}]"))
            lines.append("")

        text = "\n".join(lines).strip() or "(memory is empty — this is the first expansion)"
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n... (memory truncated)"
        return text

    # ------------------------------------------------------------------
    def write_finding(self, node_id: str, text: str) -> None:
        node = self.tree.nodes.get(node_id)
        if node is None:
            return
        snippet = text.strip()
        if not snippet:
            return
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        node.findings.append(snippet)
        self.log.append({"node_id": node_id, "finding": snippet})

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "top_k_global": self.top_k_global,
            "max_chars": self.max_chars,
            "log": list(self.log),
        }


__all__ = ["GlobalMemory"]
