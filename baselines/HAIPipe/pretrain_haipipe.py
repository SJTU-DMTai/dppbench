#!/usr/bin/env python
"""Compatibility entry point; canonical cross-task training is scripts/pretrain_baselines.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pretrain_baselines import main

if __name__ == "__main__":
    # Keep the former space-separated source/heldout lists accepted.
    incoming = sys.argv[1:]
    result = ["--baseline", "HAIPipe"]
    i = 0
    while i < len(incoming):
        key = incoming[i]
        if key in ("--data_names", "--held_out_data_names"):
            i += 1
            values = []
            while i < len(incoming) and not incoming[i].startswith("--"):
                values.append(incoming[i]); i += 1
            result.extend([key, ",".join(values)])
        else:
            result.append("--pretrained_dir" if key == "--output_dir" else key)
            i += 1
    sys.argv = [sys.argv[0], *result]
    main()
