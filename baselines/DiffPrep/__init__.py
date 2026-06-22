"""DiffPrep baseline.

Differentiable preprocessing pipeline search inspired by
*DiffPrep: Differentiable Data Preprocessing Pipeline Search for Learning over
Tabular Data* (SIGMOD'23). Adapted for the shared dppbench 58-operator zoo and
extended to support both tabular and recommendation tasks.

Public API::

    from baselines.DiffPrep.diffprep import DiffPrep
"""

from .diffprep import DiffPrep

__all__ = ["DiffPrep"]
