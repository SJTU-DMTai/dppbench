"""Auto-Prep baseline package.

Implements the SIGMOD 2024 *Auto-Prep* (Holistic Prediction of Data
Preparation Steps for Self-Service BI) workflow on top of dppbench.
"""
from __future__ import annotations

from .auto_prep import AutoPrep

__all__ = ["AutoPrep"]
