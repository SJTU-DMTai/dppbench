"""BAT sandbox (re-export of DeepPrep's sandbox).

BAT shares the same execution sandbox semantics as DeepPrep: stateful
incremental execution of operator chains via ``dppbench.dataset``,
schema/dtype/sample observation rendering, and pickle-based snapshot for
MCTS branch replay. Re-exporting avoids duplicating the implementation.
"""
from baselines.DeepPrep.sandbox import ExecutionResult, ObsState, Sandbox

__all__ = ["Sandbox", "ObsState", "ExecutionResult"]
