"""Shared FP8 weight quantization for local LLM baseline inference."""
from __future__ import annotations

import torch


def fp8_quantization_config(device: str | torch.device | None = None):
    """Quantize linear weights to FP8; keep activations and other layers in BF16."""
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 LLM loading requires a CUDA GPU")
    selected = torch.device(device or "cuda")
    if selected.type != "cuda":
        raise RuntimeError("FP8 LLM loading requires a CUDA device")
    capability = torch.cuda.get_device_capability(selected)
    if capability < (8, 9):
        raise RuntimeError(f"FP8 LLM loading requires CUDA compute capability >= 8.9; got {capability}")
    try:
        from torchao.quantization import Float8WeightOnlyConfig
        from transformers import TorchAoConfig
    except ImportError as exc:
        raise RuntimeError("FP8 LLM loading requires torchao==0.13.0") from exc
    return TorchAoConfig(Float8WeightOnlyConfig())
