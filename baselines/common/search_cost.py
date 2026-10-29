"""Observational search work accounting; never consumes an execution budget.

Counts are typed work units, NOT equivalent evaluations or FLOPs. Timings are
inclusive host wall times (no extra CUDA synchronization); nested times must
not be added. Context-local scopes isolate concurrent runs. Explicitly pass a
copied context to worker threads; subprocess aggregation is not supported.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path
import threading
import time

_ACTIVE = ContextVar('dppbench_search_cost', default=None)
_PHASE = ContextVar('dppbench_search_cost_phase', default='search')


class SearchCost:
    def __init__(self, baseline=None):
        self.baseline = baseline
        self.status = 'not_started'
        self.operations = {}
        self._lock = threading.Lock()

    def add(self, kind, **values):
        key = f'{_PHASE.get()}/{kind}'
        with self._lock:
            entry = self.operations.setdefault(key, {})
            for name, value in values.items():
                entry[name] = entry.get(name, 0) + value

    def snapshot(self):
        with self._lock:
            operations = {k: dict(v) for k, v in sorted(self.operations.items())}
        return dict(version=1, baseline=self.baseline, status=self.status,
                    timing='inclusive_host_wall_no_extra_cuda_sync',
                    coverage='common_pipeline_scoring_and_downstream_fit; '
                             'DAGPipe/DiffPrep_proxy_objectives_gradients_updates; '
                             'other_internal_models_not_instrumented',
                    operations=operations)

    def fields(self):
        if self.status == 'not_started':
            return {'search_cost': None}
        data = self.snapshot()
        ops = data['operations']

        def total(prefix, field='started'):
            return sum(v.get(field, 0) for k, v in ops.items()
                       if k.startswith('search/' + prefix))

        proxy = self.baseline in {'DAGPipe', 'DiffPrep'}
        return dict(search_cost=data,
                    search_pipeline_score_calls=total('pipeline_score.'),
                    search_downstream_fit_calls=total('downstream_fit.'),
                    search_proxy_train_loss_calls=total('proxy_train.', 'completed') if proxy else None,
                    search_proxy_val_loss_calls=total('proxy_val.', 'completed') if proxy else None,
                    search_gradient_calls=total('gradient.') if proxy else None,
                    search_optimizer_steps=total('optimizer.', 'completed') if proxy else None)


def record_work(kind, **values):
    cost = _ACTIVE.get()
    if cost is not None:
        cost.add(kind, **values)


def measured_call(kind, function, *args, **kwargs):
    """Record actual calls, including failed/partial calls, without retries."""
    cost = _ACTIVE.get()
    if cost is None:
        return function(*args, **kwargs)
    cost.add(kind, started=1)
    start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    except BaseException:
        cost.add(kind, raised=1)
        raise
    else:
        cost.add(kind, completed=1)
        if isinstance(result, dict) and 'success' in result:
            cost.add(kind, successful_results=int(bool(result['success'])))
        return result
    finally:
        cost.add(kind, host_seconds=time.perf_counter() - start)


@contextmanager
def cost_phase(phase):
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


@contextmanager
def search_cost_scope(cost, output_dir=None):
    if cost.status != 'not_started':
        raise ValueError('Use a fresh SearchCost per run')
    token = _ACTIVE.set(cost)
    phase_token = _PHASE.set('search')
    cost.status = 'running'
    start = time.perf_counter()
    try:
        yield cost
    except BaseException:
        cost.status = 'failed'
        raise
    else:
        cost.status = 'complete'
    finally:
        cost.add('scope', host_seconds=time.perf_counter() - start)
        _PHASE.reset(phase_token)
        _ACTIVE.reset(token)
        if output_dir is not None:
            path = Path(output_dir) / 'search_cost.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(cost.snapshot(), indent=2), encoding='utf-8')
            temporary.replace(path)
