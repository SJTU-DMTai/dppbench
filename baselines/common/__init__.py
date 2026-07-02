"""Shared baseline runtime utilities.

Originally lived under the top-level ``agents/`` package; moved here so
the dppbench repo only carries one location for baseline-shared code.

Public exports mirror the legacy ``agents`` package import surface for
the very small number of historic call sites (e.g.
``scripts/run_agent.py``):

- ``TrainingExecutor`` (the canonical downstream-training driver, used
  by SAGA/DeepPrep/SPIO/CtxPipe/etc. to score a pipeline yaml).
- ``PreprocOptAgent`` (a small LLM-based reference baseline that serves
  as a smoke test for the operator library).
- ``LLMCaller``
- ``get_operator_descriptions`` / ``get_operator_summary`` /
  ``get_operator_detail``
- ``Pipeline`` / ``PipelineStep`` / ``DataContext`` and shared legality /
  evaluation helpers used by all pipeline-search baselines.
"""
from .executor import TrainingExecutor
from .pipeline import (
    DataContext,
    Pipeline,
    PipelineStep,
    assign_dag_structure,
    build_default_params,
    make_step,
)
from .pipeline_constraints import ensure_tabular_tail, is_legal, repair
from .evaluator import EvaluationResult, PipelineEvaluator
from .op_registry import (
    get_operator_descriptions,
    get_operator_summary,
    get_operator_detail,
)

# Optional: PreprocOptAgent / LLMCaller depend on the optional ``openai``
# package. Import lazily to keep ``baselines.common.TrainingExecutor``
# importable on machines without ``openai`` installed.
try:  # pragma: no cover - optional dependency
    from .llm_caller import LLMCaller  # noqa: F401
    from .preproc_opt_agent import PreprocOptAgent  # noqa: F401
except Exception:  # pragma: no cover
    LLMCaller = None  # type: ignore
    PreprocOptAgent = None  # type: ignore
