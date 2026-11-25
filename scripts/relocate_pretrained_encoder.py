"""Copy a trusted handoff bundle and explicitly migrate its encoder path contract.

The caller must confirm the destination encoder is the original model. No model
identity can be inferred from the old absolute path alone. Original files remain
untouched; strict runtime checkpoint validation is unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import torch

OLD_PATH = "/mnt/cephfs/project_wx-rec-alg-bdc-exp/hachimideng/dppbench/ckpts/Qwen3-8B"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape
                and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                b.contiguous().reshape(-1).view(torch.uint8)))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def migrate(args):
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    encoder = args.encoder.expanduser().resolve(strict=True)
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a new directory outside the source tree")
    if not args.confirm_same_encoder:
        raise ValueError("Explicit --confirm-same-encoder is required")
    for name in ("config.json", "tokenizer_config.json"):
        if not (encoder / name).is_file():
            raise FileNotFoundError(encoder / name)
    indexes = list(encoder.glob("*.index.json"))
    shards = set()
    for index in indexes:
        shards.update(json.loads(index.read_text())["weight_map"].values())
    if not shards:
        shards.update(p.name for pattern in ("*.safetensors", "pytorch_model*.bin")
                      for p in encoder.glob(pattern))
    if not shards:
        raise ValueError("No encoder weights found")
    for shard in shards:
        path = (encoder / shard).resolve(strict=True)
        if encoder not in path.parents:
            raise ValueError(f"Encoder shard outside model directory: {shard}")

    # Verify the delivery before loading any trusted checkpoint.
    sums = source / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (source / name.lstrip("*")).resolve(strict=True)
        if source not in path.parents or digest(path) != expected:
            raise ValueError(f"Source checksum failed: {name}")

    records = []
    for baseline in ("FlowPipe", "CtxPipe"):
        for model in ("LightGBM", "DIN"):
            folder = source / baseline / model
            checkpoint = folder / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["checkpoint_metadata"]
            info = json.loads((folder / "pretrain.json").read_text())
            # JSON serializes tuple actions as lists.
            if json.loads(json.dumps(metadata)) != info["checkpoint_metadata"]:
                raise ValueError(f"Checkpoint / JSON contract mismatch: {folder}")
            if (metadata["baseline"] != baseline or metadata["model"] != model
                    or metadata["encoder"]["path"] != args.old_path):
                raise ValueError(f"Unexpected source contract: {folder}")
            records.append((baseline, model))
            del payload

    # Hash actual encoder files for an auditable target identity. This records
    # identity; it does not independently prove equivalence to an absent source.
    encoder_hashes = {p.relative_to(encoder).as_posix(): digest(p)
                      for p in sorted(encoder.rglob("*")) if p.is_file()}
    shutil.copytree(source, output)
    changes = []
    for baseline, model in records:
        folder = output / baseline / model
        checkpoint = folder / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        before = copy.deepcopy(payload["checkpoint_metadata"])
        payload["checkpoint_metadata"]["encoder"]["path"] = str(encoder)
        temp = checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, temp)
        reloaded = torch.load(temp, map_location="cpu", weights_only=True)
        if not equal(payload, reloaded):
            raise ValueError(f"Checkpoint round-trip failed: {checkpoint}")
        reloaded["checkpoint_metadata"] = before
        payload["checkpoint_metadata"] = before
        original = torch.load(source / baseline / model / "checkpoint.pt",
                              map_location="cpu", weights_only=True)
        if not equal(original, reloaded):
            raise ValueError(f"Unexpected non-path change: {checkpoint}")
        os.replace(temp, checkpoint)
        info_path = folder / "pretrain.json"
        info = json.loads(info_path.read_text())
        info["checkpoint_metadata"]["encoder"]["path"] = str(encoder)
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        changes.append({"file": f"{baseline}/{model}/checkpoint.pt",
                        "source_sha256": digest(source / baseline / model / "checkpoint.pt"),
                        "output_sha256": digest(checkpoint)})
        del payload, reloaded, original

    # The original manifest describes source files, not migrated artifacts.
    if (output / "manifest.json").exists():
        (output / "manifest.json").rename(output / "source_manifest.json")
    (output / "SHA256SUMS").rename(output / "SOURCE_SHA256SUMS")
    report = {"type": "explicit_encoder_path_relocation", "source": str(source),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "old_encoder_path": args.old_path, "new_encoder_path": str(encoder),
              "same_encoder_confirmed_by_user": True,
              "encoder_files_sha256": encoder_hashes, "checkpoints": changes,
              "scope": "Only checkpoint_metadata.encoder.path changed in checkpoints; "
                       "matching JSON contracts updated. HAIPipe and caches copied unchanged."}
    (output / "relocation.json").write_text(json.dumps(report, indent=2) + "\n")
    files = sorted(p for p in output.rglob("*") if p.is_file() and p != output / "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(p)}  {p.relative_to(output).as_posix()}\n" for p in files))
    print(f"Migrated bundle: {output}")
    print("Next: sha256sum -c SHA256SUMS, then run real checkpoint/data smoke tests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--old-path", default=OLD_PATH)
    parser.add_argument("--confirm-same-encoder", action="store_true")
    migrate(parser.parse_args())
