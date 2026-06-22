"""DataMaster baseline.

Implements the DataMaster paper's tree-based agentic search (DataTree +
Global Memory + UCB scheduling), with two scope adjustments per project
requirements:

* The **Red-Node / Data-Pool** branch is removed (no external data
  exploration in dppbench).
* **Black nodes** do NOT generate Python code; the LLM instead selects an
  operator chain from dppbench's shared 58 real operators, exactly the same way
  DeepPrep does. The resulting pipelines are emitted as berka-style
  ``pipeline:`` YAML.
"""
from .data_master import DataMaster

__all__ = ["DataMaster"]
