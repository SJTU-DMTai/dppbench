"""Process-local GPU visibility and local-LLM placement configuration."""
import json
import os
import re


def resolve_device(gpu_id, visible_devices=None):
    if visible_devices is not None:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Set GPU visibility before CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    if gpu_id is None or gpu_id < 0:
        return 'cpu'
    import torch
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        if torch.cuda.is_initialized():
            raise RuntimeError('Set GPU visibility before CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU requested but CUDA is unavailable; refusing silent CPU fallback')
    count = torch.cuda.device_count()
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}; visible_device_count={count}")
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        print(f'[GPU] logical cuda:{index}: {props.name}, {props.total_memory / 1024**3:.2f} GiB')
    return 'cuda:0'


def llm_placement(default_device_map='auto'):
    device_map = os.environ.get('DPPBENCH_LLM_DEVICE_MAP', default_device_map)
    if device_map not in {'auto', 'balanced', 'balanced_low_0', 'sequential', 'cpu'}:
        raise ValueError('Invalid DPPBENCH_LLM_DEVICE_MAP')
    raw = os.environ.get('DPPBENCH_LLM_MAX_MEMORY')
    kwargs = {'device_map': device_map}
    if raw:
        values = json.loads(raw)
        if not isinstance(values, dict) or not values:
            raise ValueError('LLM max_memory must be a nonempty JSON object')
        limits = {}
        for key, value in values.items():
            if key != 'cpu' and not str(key).isdigit():
                raise ValueError('max_memory keys must be logical GPU indices or cpu')
            if not isinstance(value, (int, str)) or isinstance(value, bool):
                raise ValueError('max_memory values must be bytes or sizes such as 16GiB')
            if ((isinstance(value, int) and value <= 0) or
                    (isinstance(value, str) and not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?\s*(?:[KMGT]i?B)', value, re.I))):
                raise ValueError('max_memory limits must be positive sizes, e.g. 12GiB')
            limits[int(key) if str(key).isdigit() else key] = value
        kwargs['max_memory'] = limits
    return kwargs
