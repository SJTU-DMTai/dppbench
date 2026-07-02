"""ReAct baseline: Thought -> Action -> Observation loop.

Each turn the LLM emits a complete pipeline YAML (matching the schema of
``dppbench/tasks/<task>/prepare.yaml``); the system resets the sandbox,
executes the pipeline, trains the downstream model on a small sample, and
feeds (status, schema, val-set metrics, fitness) back as ``<observation>``.
The highest-fitness attempt across all turns is kept as the final pipeline.
"""
from .react import ReAct

__all__ = ["ReAct"]
