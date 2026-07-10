"""DeepPrep baseline.

LLM-powered tree-based data-preparation agent. Exploration uses sandboxed
structural execution feedback; when configured, successful solution attempts
may also receive DPPBench downstream fitness as search feedback.
"""
from .deepprep import DeepPrep

__all__ = ["DeepPrep"]
