"""Learn2Clean baseline.

Tabular Q-learning + Boltzmann exploration over the dppbench operator zoo,
inspired by *Learn2Clean: Optimizing the Sequence of Tasks for Web Data
Preparation* (Berti-Equille, WWW'19,
https://github.com/LaureBerti/Learn2Clean).

Adapted to:

* Use the shared 58-operator catalog through
  :mod:`baselines.DiffPrep.operator_catalog` (not just data-cleaning ops).
* Score candidate pipelines via the real downstream model AUC -- the Learn2Clean
  reward is the *step-wise AUC delta* plus a terminal AUC delta, matching the
  paper's "min-max-normalised quality difference" reward shape.
* Support both tabular and recommendation tasks.

Public API::

    from baselines.Learn2Clean.learn2clean import Learn2Clean
"""

from .learn2clean import Learn2Clean

__all__ = ["Learn2Clean"]
