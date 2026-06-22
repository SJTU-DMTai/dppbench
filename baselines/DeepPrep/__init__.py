"""DeepPrep baseline.

LLM-powered agentic pipeline construction with tree-based reasoning.
The agent does NOT receive any downstream-model feedback; it only sees
sandboxed operator-execution outcomes (schema / sample / errors).
"""
from .deepprep import DeepPrep

__all__ = ["DeepPrep"]
