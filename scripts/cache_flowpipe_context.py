"""Precompute and persist FlowPipe's frozen Llama semantic embeddings."""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baselines.FlowPipe.flowpipe import FlowPipe  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_names", required=True, help="Comma-separated DPPBench tasks")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--llama_model_path",
        default="/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B",
    )
    parser.add_argument("--llm_name", default="Qwen3-8B")
    parser.add_argument("--ctx_dim", type=int, default=4096)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _task_data_dir(data_root: str, data_name: str) -> str:
    return os.path.join(os.path.abspath(data_root), data_name, "data")


def main() -> int:
    args = parse_args()
    device = f"cuda:{args.gpu_id}"
    cache_dir = os.path.join(os.path.abspath(args.output_dir), "FlowPipe", "context_embeddings")
    for data_name in [name.strip() for name in args.data_names.split(",") if name.strip()]:
        task_dir = os.path.join(_ROOT, "dppbench", "tasks", data_name)
        baseline = FlowPipe(
            task_dir=task_dir, data_name=data_name,
            data_dir=_task_data_dir(args.data_dir, data_name),
            output_dir=os.path.join(args.output_dir, "FlowPipe", data_name),
            context_cache_dir=cache_dir,
            llama_model_path=args.llama_model_path,
            llm_name=args.llm_name, ctx_dim=args.ctx_dim,
            device=device, verbose=True,
        )
        try:
            path = baseline.precompute_context(
                device=device, max_length=args.max_length,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            if "graph" in str(exc).lower():
                print(f"[{data_name}] skipped: {exc}")
                continue
            raise
        print(f"[{data_name}] saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
