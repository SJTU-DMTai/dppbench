"""Compatibility wrapper for shared pipeline utilities.

The implementation lives in :mod:`baselines.common.pipeline` because these
classes and factories are used by multiple baselines, not just SAGA.
"""
from baselines.common.pipeline import *  # noqa: F401,F403
