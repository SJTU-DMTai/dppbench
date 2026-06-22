"""UCB scheduling for the DataMaster baseline.

Implements paper Eq.(2):

    Score(v) = R_v / N_v + c_t * sqrt( log N_Par(v) / N_v )

with a decaying exploration coefficient ``c_t`` (linear / exponential /
piecewise / none) — as described in §3.5. ``select`` consumes a
:class:`DataTree` frontier and returns the next node to expand.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .data_tree import DataTree, NodeRecord


@dataclass
class UCBSchedulerConfig:
    c_initial: float = 1.414
    c_lower_bound: float = 0.1
    decay: str = "linear"          # "linear" | "exponential" | "piecewise" | "none"
    decay_alpha: float = 0.05      # used by linear
    decay_gamma: float = 0.95      # used by exponential
    piecewise_t1: int = 5
    piecewise_t2: int = 20
    reward_kind: str = "fitness"   # "fitness" | "improvement"


class UCBScheduler:
    """UCB-based frontier selector with a decaying ``c_t`` schedule."""

    def __init__(self, config: Optional[UCBSchedulerConfig] = None) -> None:
        self.config = config or UCBSchedulerConfig()

    # ------------------------------------------------------------------
    def current_c(self, step: int) -> float:
        cfg = self.config
        kind = cfg.decay
        if kind == "none":
            c = cfg.c_initial
        elif kind == "linear":
            c = cfg.c_initial - cfg.decay_alpha * step
        elif kind == "exponential":
            c = cfg.c_initial * (cfg.decay_gamma ** max(0, step))
        elif kind == "piecewise":
            if step < cfg.piecewise_t1:
                c = cfg.c_initial
            elif step < cfg.piecewise_t2:
                c = cfg.c_initial * 0.5
            else:
                c = cfg.c_initial * 0.25
        else:
            raise ValueError(f"Unknown decay kind {kind!r}")
        return max(cfg.c_lower_bound, c)

    # ------------------------------------------------------------------
    def score(
        self, node: NodeRecord, parent: Optional[NodeRecord], step: int
    ) -> float:
        """Compute UCB(v). Unvisited nodes return +inf (sample first)."""
        if node.visits == 0:
            return math.inf
        n_par = parent.visits if parent is not None else node.visits
        n_par = max(1, n_par)
        c_t = self.current_c(step)
        exploit = node.total_reward / node.visits
        explore = c_t * math.sqrt(math.log(n_par) / node.visits)
        return exploit + explore

    # ------------------------------------------------------------------
    def select(self, tree: DataTree, step: int) -> Optional[NodeRecord]:
        """Pick the highest-scoring frontier node, breaking ties by depth
        (prefer deeper) then by recency (older first, to give every branch
        a chance).
        """
        frontier = tree.frontier()
        if not frontier:
            return None
        best: Optional[NodeRecord] = None
        best_score = -math.inf
        for node in frontier:
            parent = tree.parent(node.node_id)
            sc = self.score(node, parent, step)
            if sc > best_score or (
                sc == best_score and best is not None
                and (node.depth > best.depth or
                     (node.depth == best.depth and node.created_at < best.created_at))
            ):
                best = node
                best_score = sc
        return best

    # ------------------------------------------------------------------
    def compute_reward(
        self, child: NodeRecord, parent: Optional[NodeRecord]
    ) -> float:
        """Map an evaluated child node to a scalar reward.

        * ``fitness``: pass through ``y_v`` (None -> 0). Easy to interpret
          and matches the paper's UCB formula.
        * ``improvement``: ML-Master-style ±1/0 signal vs. parent fitness;
          unsuccessful evaluations get -1.
        """
        kind = self.config.reward_kind
        if kind == "fitness":
            return float(child.fitness) if child.fitness is not None else 0.0
        if kind == "improvement":
            if child.fitness is None:
                return -1.0
            par_fit = parent.fitness if (parent is not None and parent.fitness is not None) else None
            if par_fit is None:
                # Compare against root proxy: any successful black node is better than nothing
                return 1.0
            if child.fitness > par_fit:
                return 1.0
            if child.fitness < par_fit:
                return -1.0
            return 0.0
        raise ValueError(f"Unknown reward_kind {kind!r}")


__all__ = ["UCBScheduler", "UCBSchedulerConfig"]
