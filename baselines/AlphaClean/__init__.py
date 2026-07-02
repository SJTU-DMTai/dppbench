"""AlphaClean baseline.

Best-first beam search over the dppbench operator zoo, inspired by
*AlphaClean: Automatic Generation of Data Cleaning Pipelines* (Krishnan & Wu,
VLDB'19). Adapted to:

* Use the shared operator catalog through
  :mod:`baselines.DiffPrep.operator_catalog` (not just data-cleaning ops).
* Score candidate pipelines via the real downstream model AUC, matching the
  evaluator contract used by SAGA / CtxPipe / DiffPrep.
* Support both tabular and recommendation tasks.

Public API::

    from baselines.AlphaClean.alphaclean import AlphaClean
"""

from .alphaclean import AlphaClean

__all__ = ["AlphaClean"]
