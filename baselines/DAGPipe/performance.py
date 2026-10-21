"""Bounded profiling and conservative, runtime-aware tensor placement."""
from pathlib import Path
from contextlib import contextmanager, nullcontext
import time

import torch


def clip_gradients(parameters, check=True, instrument=False):
    """L2 clip at one; keep the float64 norm and coefficient on the device."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None
    with torch.profiler.record_function('dagpipe.gradient_norm') if instrument else nullcontext():
        norm = torch.stack([torch.linalg.vector_norm(g.double()) for g in grads]).norm()
    finite = torch.isfinite(norm)
    if check and not bool(finite):
        raise RuntimeError('DAGPipe encountered non-finite gradients or gradient norm')
    factor = (1.0 / (norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(factor.to(grad.dtype))
    return finite


def legacy_clip_gradients(parameters):
    """Measurement-only reference path, matching the pre-optimization syncs."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('DAGPipe encountered non-finite gradients')
    norm = torch.stack([torch.linalg.vector_norm(g.double()) for g in grads]).norm()
    if not torch.isfinite(norm):
        raise RuntimeError('DAGPipe encountered non-finite gradient norm')
    factor = min(1., 1./(norm.item()+1e-6))
    for grad in grads:
        grad.mul_(factor)


def placement_plan(tensors, free_bytes, total_bytes, reserve_mb=4096, fraction=.5,
                   budget_remaining_bytes=None, future_optimizer_bytes=0):
    """Reserve workspace AND use at most a fraction of currently free VRAM.

    Includes an int64 permutation on GPU; optimizer/activation headroom is an
    explicit conservative reserve, not an assertion of exact peak memory.
    """
    if not 0 < fraction <= 1 or reserve_mb < 0:
        raise ValueError('Invalid GPU memory reserve/fraction')
    tensor_bytes = sum(t.numel()*t.element_size() for t in tensors)
    permutation_bytes = len(tensors[0])*8
    required = tensor_bytes + permutation_bytes + future_optimizer_bytes
    available = free_bytes if budget_remaining_bytes is None else min(free_bytes, budget_remaining_bytes)
    capacity = max(0, min(int(available*fraction), available-int(reserve_mb*1024**2)))
    return dict(tensor_bytes=tensor_bytes, permutation_bytes=permutation_bytes,
                required_bytes=required, free_bytes=free_bytes, total_bytes=total_bytes,
                reserve_mb=reserve_mb, future_optimizer_bytes=future_optimizer_bytes,
                capacity_bytes=capacity, fits=required <= capacity)


class TrainingData:
    def __init__(self, tensors, device, cfg, budget=None, parameters=()):
        self.device = torch.device(device)
        self.tensors = tuple(tensors)
        self.resident = False
        mode = cfg.get('data_placement', 'auto')
        if mode not in ('auto', 'cpu'):
            raise ValueError('data_placement must be auto or cpu')
        self.report = dict(mode='cpu', device=str(self.device))
        if self.device.type != 'cuda':
            return
        free, total = torch.cuda.mem_get_info(self.device)
        remaining = None
        if budget and budget.max_gpu_memory_mb:
            remaining = int(budget.max_gpu_memory_mb*1024**2)-torch.cuda.memory_reserved(self.device)
        self.report.update(placement_plan(tensors, free, total,
                           float(cfg.get('gpu_reserve_mb', 4096)),
                           float(cfg.get('gpu_data_fraction', .5)), remaining,
                           sum(p.numel()*p.element_size()*3 for p in parameters)))
        if mode == 'auto' and self.report['fits']:
            moved = []
            try:
                for tensor in tensors:
                    moved.append(tensor.to(self.device))
                self.tensors = tuple(moved)
                self.resident = True
                self.report['mode'] = 'cuda_resident'
            except torch.cuda.OutOfMemoryError:
                moved.clear()
                torch.cuda.empty_cache()
                self.report['fallback'] = 'allocation_oom'
        if not self.resident:
            self.report['mode'] = 'cpu_pinned_batches'

    def indices(self, indices):
        # Preserve the CPU generator's permutation / random trajectory.
        return indices.to(self.device) if self.resident else indices

    def batch(self, validation, index):
        x, y = self.tensors[2:] if validation else self.tensors[:2]
        values = (x[index], y[index])
        if self.resident or self.device.type != 'cuda':
            return values
        # Bound pinned host memory to a batch; never pin millions of rows.
        return tuple(v.pin_memory().to(self.device, non_blocking=True) for v in values)


class ProfileWindow:
    """Only instrument the requested first N optimizer steps; disabled by default."""
    def __init__(self, steps, device, output):
        self.remaining = int(steps)
        self.profiler = None
        self.output = Path(output)
        if self.remaining > 0:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.device(device).type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities, record_shapes=True,
                                                   profile_memory=True)
            self.profiler.start()

    def step(self):
        if self.profiler:
            self.profiler.step()
            self.remaining -= 1
            if self.remaining <= 0:
                self.close()

    def close(self):
        if self.profiler:
            profiler, self.profiler = self.profiler, None
            profiler.stop()
            self.output.mkdir(parents=True, exist_ok=True)
            trace = self.output/'training_trace.json'
            profiler.export_chrome_trace(trace.as_posix())
            # Older Windows Kineto may write a valid .tmp trace but fail its
            # native rename. Complete it with Python and verify the artifact.
            temporary = Path(str(trace)+'.tmp')
            if not trace.exists() and temporary.exists():
                temporary.replace(trace)
            if not trace.exists():
                raise RuntimeError(f'Profiler did not export a trace: {trace}')
            (self.output/'training_profile.txt').write_text(
                profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=40), encoding='utf-8')
            if any(a == torch.profiler.ProfilerActivity.CUDA for a in profiler.activities):
                (self.output/'training_profile_cuda.txt').write_text(
                    profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=40), encoding='utf-8')


class ResourceGuard:
    """Throttle expensive RSS/device queries, keeping mandatory boundary checks."""
    def __init__(self, check, interval=.1):
        if interval < 0:
            raise ValueError('resource_check_interval_s must be nonnegative')
        self.check, self.interval, self.last = check, interval, float('-inf')

    def __call__(self, force=False):
        now = time.monotonic()
        if force or now-self.last >= self.interval:
            self.check()
            self.last = now


class StepMeasurements:
    """Diagnostic ranges: host wall time and stream event elapsed time.

    Events may include stream idle/host gaps; they are not sums of GPU kernel
    execution times. Use the operator profiler for that attribution.
    """
    def __init__(self, device, enabled=False):
        self.device, self.enabled = torch.device(device), enabled
        self.rows = []
        self.events = []

    def scope(self, name):
        return self._scope(name) if self.enabled else nullcontext()

    @contextmanager
    def _scope(self, name):
        events = None
        if self.device.type == 'cuda':
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        with torch.profiler.record_function('dagpipe.'+name):
            yield
        self.rows.append((name, time.perf_counter()-started))
        if events:
            events[1].record(torch.cuda.current_stream(self.device))
            self.events.append((name, *events))

    def summary(self):
        totals = {}
        for name, seconds in self.rows:
            item = totals.setdefault(name, {'host_wall_seconds': 0.0, 'calls': 0})
            item['host_wall_seconds'] += seconds
            item['calls'] += 1
        if self.events:
            torch.cuda.synchronize(self.device)
            for name, start, end in self.events:
                totals[name]['cuda_stream_elapsed_ms'] = totals[name].get('cuda_stream_elapsed_ms', 0)+start.elapsed_time(end)
        return totals
