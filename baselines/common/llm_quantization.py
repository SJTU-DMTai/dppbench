"""Shared FP8 weight quantization for local LLM baseline inference."""
from __future__ import annotations

import os
import warnings
import torch


def fp8_quantization_config(device: str | torch.device | None = None):
    """DPPBENCH_LLM_QUANTIZATION: auto (default), none, or strict fp8."""
    mode = os.environ.get('DPPBENCH_LLM_QUANTIZATION', 'auto').lower()
    if mode not in {'auto', 'none', 'fp8'}:
        raise ValueError('DPPBENCH_LLM_QUANTIZATION must be auto, none, or fp8')
    if mode == 'none':
        return None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('FP8 requires CUDA')
        selected = torch.device('cuda' if device in (None, 'auto') else device)
        if selected.type != 'cuda':
            raise RuntimeError('FP8 requires a CUDA device')
        indices = range(torch.cuda.device_count()) if device in (None, 'auto') else [selected]
        if any(torch.cuda.get_device_capability(index) < (8, 9) for index in indices):
            raise RuntimeError('FP8 requires CUDA compute capability >= 8.9')
        from torchao.quantization import Float8WeightOnlyConfig
        from transformers import TorchAoConfig
        return TorchAoConfig(Float8WeightOnlyConfig())
    except (ImportError, RuntimeError, AttributeError, OSError) as exc:
        if mode == 'fp8':
            raise RuntimeError(f'Explicit FP8 loading is unavailable: {exc}') from exc
        warnings.warn(f'FP8 unavailable; loading unquantized weights: {exc}', RuntimeWarning)
        return None


def local_torch_dtype(device=None):
    """FP32 on CPU, BF16 on supported CUDA hardware, otherwise FP16."""
    selected = torch.device('cuda' if device in (None, 'auto') and torch.cuda.is_available()
                            else 'cpu' if device in (None, 'auto') else device)
    if selected.type != 'cuda':
        return torch.float32
    major, _ = torch.cuda.get_device_capability(selected)
    return torch.bfloat16 if major >= 8 else torch.float16


class LocalModelInitializationError(RuntimeError):
    """A local model cannot load; retrying search actions cannot fix this."""


def require_accelerate(device_map):
    """Use Transformers' own version-aware dependency check."""
    if device_map is None:
        return
    import sys
    from transformers.utils import is_accelerate_available
    if not is_accelerate_available():
        from transformers.utils.import_utils import ACCELERATE_MIN_VERSION
        raise LocalModelInitializationError(
            f"Local model device_map={device_map!r} requires accelerate>={ACCELERATE_MIN_VERSION}. "
            f"Install into the running environment: {sys.executable} -m pip install "
            f"'accelerate>={ACCELERATE_MIN_VERSION}', then restart the experiment process."
        )


def model_dtype_kwargs(dtype):
    """Use the installed Transformers loading API, including older releases."""
    import transformers
    from packaging.version import Version
    # Both keywords are consumed through **kwargs, not the public signature.
    key = 'dtype' if Version(transformers.__version__).release >= (4, 56) else 'torch_dtype'
    return {key: dtype}
