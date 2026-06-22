"""BAT baseline package.

Implements the *Target-Instance-Free* LLM-driven Monte Carlo Tree Search
data-preparation synthesizer (BAT, ZJU-DAILY/BAT) on top of dppbench's 58
operators, with an additional downstream-model feedback channel.
"""
from .bat import BAT

__all__ = ["BAT"]
