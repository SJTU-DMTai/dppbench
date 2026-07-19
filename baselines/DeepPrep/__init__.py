"""DeepPrep baseline.

LLM-powered tree-based data-preparation agent using the released protocol:
model-generated plan/expand/answer turns and environment-generated execute
feedback. DPPBench downstream metrics are computed only after inference.
"""
from .deepprep import DeepPrep

__all__ = ["DeepPrep"]
