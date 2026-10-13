"""Sandbox for the DeepPrep agent.

Materializes intermediate table states, runs operator chains via DPPBench's
prev-only DAG YAML, and produces tag-friendly observation text containing
schema, dtype counts, and sample rows. Crucially it never trains a downstream
model — its execution feedback is purely structural.

Snapshot / restore is implemented with ``pickle.dumps``/``pickle.loads`` over
the underlying ``BaseData`` object, allowing the tree agent to backtrack to
any previously visited node.
"""
from __future__ import annotations

import os
import pickle
import random as _random
import shutil
import sys
import tempfile
import traceback
import yaml
from dataclasses import dataclass, field
from typing import Optional

from baselines.common.pipeline import (
    Pipeline,
    PipelineStep,
    assign_dag_structure,
    check_pipeline,
)
from baselines.common.search_evaluator import subsample_data_inplace
from baselines.common.budget import BudgetExceededError
from baselines.common.utils import get_data_ctx


# ---------------------------------------------------------------------------
# TrainingExecutor lives in ``baselines.common.executor``. Make sure the repo
# root is importable so the package resolves regardless of invocation style.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common.executor import TrainingExecutor  # noqa: E402


@dataclass
class ObsState:
    """Structured representation of a sandbox state."""

    shape: tuple
    columns: list[str]
    dtypes: dict
    head: list[dict]
    null_ratio: float
    text: str
    tables: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    obs: Optional[ObsState] = None
    error: Optional[str] = None
    snapshot: Optional[bytes] = None
    applied_ops: list[str] = field(default_factory=list)


class Sandbox:
    """Stateful sandbox for incremental operator-chain execution.

    The sandbox owns a mutable ``data`` instance produced by the dataset's
    ``Data`` class. Subsampling reuses the common search-time small-N helper
    so its behavior stays aligned with ``SearchPipelineEvaluator``.
    """

    SAMPLE_ROWS = 5
    MAX_COLS_IN_TEXT = 20

    def __init__(
        self,
        task_dir: str,
        data_name: str,
        data_dir: Optional[str] = None,
        small_n: int = 0,
        seed: int = 42,
    ) -> None:
        self.task_dir = task_dir
        self.data_name = data_name
        self.data_dir = data_dir
        self.small_n = int(small_n) if small_n else 0
        self.seed = int(seed)
        self._executor = TrainingExecutor(task_dir, data_name=data_name, data_dir=data_dir)
        self._tmpdir = tempfile.mkdtemp(prefix="deepprep_sandbox_")
        self.task_type: str = self._executor.task_type
        self.data = None  # populated by reset()
        self._ctx = None
        self._initial_snapshot: Optional[bytes] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> ExecutionResult:
        """(Re)load the original data and return the initial observation."""
        data = self._executor._make_configured_data_instance()
        data.load_data()
        # run_pre_process() loads frozen std-test lazily and may replace the
        # sampled training frame with the full frozen frame. Load it first so
        # the sandbox's small-N sample is the frame actually executed.
        data.load_std_test_frozen()
        if self.small_n > 0:
            subsample_data_inplace(data, small_n=self.small_n, seed=self.seed)
        # Fix train/validation partitions before any tree edge (also for RecData).
        prepare = getattr(data, "_prepare_validation", None)
        if callable(prepare):
            prepare()
        self.data = data
        self._ctx = get_data_ctx(self.data_name, self.data, self.task_type)
        self._initial_snapshot = self.snapshot()
        obs = self._observe()
        return ExecutionResult(success=True, obs=obs, snapshot=self._initial_snapshot)

    def cleanup(self) -> None:
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self) -> bytes:
        if self.data is None:
            raise RuntimeError("Sandbox.snapshot() called before reset().")
        return pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, snap: bytes) -> ObsState:
        self.data = pickle.loads(snap)
        return self._observe()

    def _context(self):
        if self.data is None:
            raise RuntimeError("Sandbox context requested before reset().")
        if self._ctx is None:
            self._ctx = get_data_ctx(self.data_name, self.data, self.task_type)
        return self._ctx

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute_chain(self, steps: list[PipelineStep]) -> ExecutionResult:
        """Apply ``steps`` to the current data state using common DAG APIs.

        On failure, the data state is rolled back to what it was before this
        call. Callers that want operator-level tree nodes should call this with
        one step at a time.
        """
        if self.data is None:
            raise RuntimeError("Sandbox.execute_chain() called before reset().")
        if not steps:
            return ExecutionResult(success=True, obs=self._observe(), applied_ops=[])

        pre_snapshot = self.snapshot()
        pipe = Pipeline(steps=[PipelineStep.from_dict(s.to_dict()) for s in steps])
        ctx = self._context()
        try:
            assign_dag_structure(pipe, ctx, _random.Random(self.seed))
            # This DAG executes only the next edge, not the final pipeline.
            # Keep referenced sources; auxiliary tables remain in sandbox data
            # and can be joined by a later edge. Final validation uses all sources.
            referenced = set(pipe.train_prev or [])
            for step in pipe.steps:
                referenced.update(step.prev)
            pipe.sources = [s for s in pipe.sources or [] if s['id'] in referenced]
            report = check_pipeline(pipe, ctx, require_mandatory=False)
            if not report.valid:
                self.data = pickle.loads(pre_snapshot)
                return ExecutionResult(
                    success=False,
                    error="Pipeline precheck failed: " + "; ".join(report.errors),
                    applied_ops=[],
                )
            chain_yaml = pipe.to_yaml()
            chain_path = os.path.join(self._tmpdir, "chain.yaml")
            with open(chain_path, "w", encoding="utf-8") as f:
                f.write(chain_yaml)
            if ctx.task_type == "rec":
                # RecData.run_pre_process finalizes by remapping IDs. Doing
                # that after every tree edge changes join keys prematurely.
                # Execute the same shared DAG/transform logic without training
                # finalization; final benchmark replay still uses run_pre_process.
                self.data.interaction_df = self.data._run_relation_dag(
                    yaml.safe_load(chain_yaml), self.data._relation_dag_available_sources(),
                    self.data._transform_rec_dag_df,
                )
            else:
                self.data.run_pre_process(chain_path)
        except BudgetExceededError:
            self.data = pickle.loads(pre_snapshot)
            raise
        except Exception as e:
            err = (
                f"{type(e).__name__}: {e}\n"
                f"{traceback.format_exc()[-400:]}"
            )
            err = err.replace(self._tmpdir, "<sandbox>")
            self.data = pickle.loads(pre_snapshot)
            return ExecutionResult(success=False, error=err, applied_ops=[])

        applied = [s.op for s in steps]
        return ExecutionResult(
            success=True,
            obs=self._observe(),
            snapshot=self.snapshot(),
            applied_ops=applied,
        )

    # ------------------------------------------------------------------
    # Observation rendering
    # ------------------------------------------------------------------
    def _primary_df(self):
        if self.task_type == "tabular":
            return getattr(self.data, "train_df", None)
        return getattr(self.data, "interaction_df", None)

    def _observe(self) -> ObsState:
        df = self._primary_df()
        if df is None:
            return ObsState(
                shape=(0, 0), columns=[], dtypes={}, head=[],
                null_ratio=0.0, text="(no primary table available)", tables={},
            )
        df = self._visible_frame(df)
        cols = [str(c) for c in df.columns]
        dtypes = {c: str(df[c].dtype) for c in cols}
        try:
            null_ratio = float(df.isnull().mean().mean())
        except Exception:
            null_ratio = 0.0
        try:
            head = df.head(self.SAMPLE_ROWS).to_dict(orient="records")
            head = [
                {k: _truncate_value(v) for k, v in row.items()}
                for row in head
            ]
        except Exception:
            head = []
        tables = self._collect_tables()
        text = self._render_text(tables)
        return ObsState(
            shape=tuple(df.shape),
            columns=cols,
            dtypes=dtypes,
            head=head,
            null_ratio=null_ratio,
            text=text,
            tables=tables,
        )

    def _collect_tables(self) -> dict:
        named = {}
        if self.task_type == "tabular":
            named["main"] = getattr(self.data, "train_df", None)
            named.update(getattr(self.data, "auxiliary_dfs", {}) or {})
        else:
            named["interaction"] = getattr(self.data, "interaction_df", None)
            named["user"] = getattr(self.data, "user_df", None)
            named["item"] = getattr(self.data, "item_df", None)
        result = {}
        for name, table in named.items():
            if table is None or not hasattr(table, "columns"):
                continue
            table = self._visible_frame(table)
            cols = [str(c) for c in table.columns]
            dtypes = {c: str(table[c].dtype) for c in cols}
            try:
                null_ratio = float(table.isnull().mean().mean())
            except Exception:
                null_ratio = 0.0
            try:
                rows = table.head(self.SAMPLE_ROWS).to_dict(orient="records")
                rows = [{k: _truncate_value(v) for k, v in row.items()} for row in rows]
            except Exception:
                rows = []
            result[str(name)] = {
                "shape": tuple(table.shape),
                "columns": cols,
                "dtypes": dtypes,
                "null_ratio": null_ratio,
                "head": rows,
            }
        return result

    @staticmethod
    def _visible_frame(frame):
        """Do not expose frozen-test statistics or examples to the LLM."""
        if "__split__" in frame.columns:
            frame = frame.loc[frame["__split__"] != "std_test"]
        return frame.drop(columns="__split__", errors="ignore")

    def _render_text(self, tables: dict) -> str:
        lines = [f"materialized_tables: {list(tables)}"]
        for name, profile in tables.items():
            cols = profile["columns"]
            col_view = cols[: self.MAX_COLS_IN_TEXT]
            more = "" if len(cols) <= self.MAX_COLS_IN_TEXT else f" (+{len(cols) - self.MAX_COLS_IN_TEXT} more)"
            n_rows, n_cols = profile["shape"]
            lines.append(f"table {name}: rows={n_rows} cols={n_cols} null_ratio_mean={profile['null_ratio']:.4f}")
            lines.append(f"  columns ({len(cols)}){more}:")
            for col in col_view:
                lines.append(f"    - {col}: {profile['dtypes'][col]}")
            categorical = [c for c in cols if profile['dtypes'][c] in ('object', 'string', 'category')]
            lines.append(f"  categorical_columns: {categorical}")
            rows = [{c: row[c] for c in col_view if c in row} for row in profile['head']]
            lines.append(f"  sample_rows (displayed columns only): {rows}")
        return "\n".join(lines)


def _truncate_value(v, limit: int = 80):
    s = repr(v)
    if len(s) > limit:
        return s[:limit] + "..."
    return v


__all__ = ["Sandbox", "ObsState", "ExecutionResult"]
