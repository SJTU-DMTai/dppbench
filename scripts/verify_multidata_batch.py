"""Read-only pre/post checks for original-protocol missingness batches."""
import argparse
import json
from pathlib import Path

import prepare_multidata_missingness as base


APPROVED = {
    "baselines/CtxPipe/env.py", "baselines/CtxPipe/config.yaml",
    "baselines/CtxPipe/context.py", "baselines/CtxPipe/trainer.py",
    "baselines/FlowPipe/config.yaml", "baselines/DAGPipe/config.yaml",
    "baselines/HAIPipe/config.yaml", "scripts/run_full_baseline_experiments.py",
}


def verify(root):
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Incomplete preparation")
    manifest = json.loads((root / "missingness_manifest.json").read_text())
    if Path(manifest["repo"]).resolve() != base.REPO.resolve():
        raise ValueError("Run from the checkout recorded in the manifest")
    if manifest["version"] not in {"original_protocol_primary_MCAR_v1",
                                    "original_protocol_primary_conditional_v1"}:
        return base.verify(root)
    drift = []
    for rel, expected in manifest["code_hashes"].items():
        path = base.REPO / rel
        actual = base.sha(path) if path.is_file() else None
        if actual != expected:
            if actual is not None and Path(rel).as_posix() in APPROVED:
                print(f"[ACCEPTED DRIFT] {rel}: {expected} -> {actual}", flush=True)
            else:
                drift.append(rel)
    if drift:
        raise ValueError("Other tracked code/config changed:\n" + "\n".join(drift))
    for rel, expected in manifest["input_hashes"].items():
        if base.sha(root / rel) != expected:
            raise ValueError(f"Copied input changed: {rel}")
    for task in manifest["tasks"]:
        hashes = json.loads((root / task / "audit/source_hashes.json").read_text())
        for path, expected in hashes.items():
            if base.sha(path) != expected:
                raise ValueError(f"Original input changed: {path}")
    print("[VERIFIED] Original inputs, copied inputs and in-scope code/config", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    verify(parser.parse_args().output_root.resolve())
