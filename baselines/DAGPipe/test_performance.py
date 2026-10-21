"""Numerical contracts for DAGPipe performance changes."""
import copy

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch

from baselines.DAGPipe.runtime import make_model, reference_forward
from baselines.DAGPipe.vendor.acyclicity import acyclicity_constraint
from baselines.DAGPipe.performance import clip_gradients, legacy_clip_gradients, placement_plan, TrainingData
from baselines.DAGPipe.performance import ResourceGuard
from baselines.DAGPipe.runtime import InputEncoder, FrozenTransform
from baselines.DAGPipe.pipeline import DAGPipePipeline


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_clipping_matches_legacy_and_rejects_nonfinite(dtype):
    left = [torch.nn.Parameter(torch.randn(4, 5, dtype=dtype)) for _ in range(3)]
    right = copy.deepcopy(left)
    for a, b in zip(left, right):
        a.grad = torch.randn_like(a)*30
        b.grad = a.grad.clone()
    legacy_clip_gradients(left)
    clip_gradients(right)
    for a, b in zip(left, right):
        torch.testing.assert_close(a.grad, b.grad)
    right[0].grad[0, 0] = float('inf')
    with pytest.raises(RuntimeError, match='non-finite'):
        clip_gradients(right)


def test_memory_plan_respects_headroom_budget_and_dtype():
    values = [torch.zeros(100, 26, dtype=torch.float64), torch.zeros(100, dtype=torch.long),
              torch.zeros(10, 26, dtype=torch.float64), torch.zeros(10, dtype=torch.long)]
    plan = placement_plan(values, 20*1024**3, 24*1024**3)
    assert plan['required_bytes'] == 110*27*8 + 100*8
    assert plan['fits']
    assert not placement_plan(values, 3*1024**3, 24*1024**3)['fits']
    assert not placement_plan(values, 20*1024**3, 24*1024**3,
                              budget_remaining_bytes=1024)['fits']
    assert placement_plan(values, 20*1024**3, 24*1024**3,
                          future_optimizer_bytes=300)['required_bytes'] == plan['required_bytes']+300
    with pytest.raises(ValueError):
        placement_plan(values, 100, 100, fraction=2)


def test_cpu_batches_preserve_order_and_validation():
    x, y = torch.arange(30).reshape(10, 3), torch.arange(10)
    data = TrainingData((x, y, x[:2], y[:2]), 'cpu', {})
    indices = torch.tensor([5, 1, 9])
    got_x, got_y = data.batch(False, data.indices(indices))
    torch.testing.assert_close(got_x, x[indices])
    torch.testing.assert_close(got_y, y[indices])
    torch.testing.assert_close(data.batch(True, slice(0, 2))[0], x[:2])


def test_resident_allocation_oom_preserves_host_data(monkeypatch):
    values = (torch.zeros(10, 2), torch.zeros(10), torch.zeros(2, 2), torch.zeros(2))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (20*1024**3, 24*1024**3))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    def oom(self, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError('simulated resident allocation')
    monkeypatch.setattr(torch.Tensor, 'to', oom)
    data = TrainingData(values, 'cuda:0', {})
    assert not data.resident and data.report['fallback'] == 'allocation_oom'
    assert all(a is b for a, b in zip(data.tensors, values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires remote CUDA device')
def test_cuda_resident_and_streamed_batches_match():
    x, y = torch.randn(20, 4, dtype=torch.float64), torch.arange(20)
    indices = torch.tensor([9, 1, 15])
    for mode in ('auto', 'cpu'):
        data = TrainingData((x, y, x[:3], y[:3]), 'cuda:0', {'data_placement': mode})
        a, b = data.batch(False, data.indices(indices))
        torch.testing.assert_close(a.cpu(), x[indices])
        torch.testing.assert_close(b.cpu(), y[indices])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('size', [1, 2, 5, 10])
def test_upper_triangular_penalty_has_zero_value_and_gradient(dtype, size):
    torch.manual_seed(11)
    logits = torch.randn(size, size, dtype=dtype, requires_grad=True)
    adjacency = torch.triu(logits.sigmoid(), diagonal=1)
    value = acyclicity_constraint(adjacency)
    gradient, = torch.autograd.grad(value, logits)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-6, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)


def test_removing_acyclicity_preserves_adam_step():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    a = make_model([['Identity', 'StandardScaler'], ['Identity', 'MinMaxScaler']], 3)
    b = copy.deepcopy(a)
    inputs = torch.randn(16, 3, dtype=torch.float64)
    reference = torch.randn(8, 3, dtype=torch.float64)
    target = torch.randn_like(inputs)
    for model, include_penalty in [(a, True), (b, False)]:
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = (reference_forward(model, inputs, reference)-target).square().mean()
        if include_penalty:
            matrices = [model.layer_adj_logits, *model.intra_adj_logits]
            loss = loss + .1 * sum(acyclicity_constraint(torch.triu(m.sigmoid(), diagonal=1)) for m in matrices)
        loss.backward()
        optimizer.step()
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def frozen_fixture(cache_bytes=128):
    frame = pd.DataFrame({'a': np.arange(8.), 'b': np.arange(8.)**2})
    encoder = InputEncoder().fit(frame)
    layers = [['Identity', 'StandardScaler']]
    model = make_model(layers, 2)
    payload = dict(format='dppbench.DAGPipe.reference.v2', task='fixture', seed=42,
                   encoder=encoder.state, layers=layers, dtype='float64',
                   state_dict={k: v.tolist() for k, v in model.state_dict().items()},
                   reference=encoder.transform(frame.head(3)).tolist(), cache_entries=3,
                   cache_max_mb=cache_bytes/1024**2)
    return payload, frame


def test_cache_bytes_lru_and_oversize_without_extra_copy():
    payload, frame = frozen_fixture()
    frozen = FrozenTransform(payload)
    # Each 3x2 float64 result is 48 bytes; 3 results exceed the 128-byte cap.
    for start in range(3):
        frozen.transform(frame.iloc[start:start+3])
    assert frozen.stats['evictions'] == 1 and frozen.cache_bytes == 96
    assert frozen.stats['peak_cached_bytes'] <= 128
    frozen.transform(pd.concat([frame, frame]))
    assert frozen.stats['oversize_skips'] == 1 and frozen.cache_bytes == 96
    assert sum(value.nbytes for value in frozen.cache.values()) == frozen.cache_bytes


def test_cache_key_contents_order_and_caller_mutation():
    payload, frame = frozen_fixture(cache_bytes=4096)
    frozen = FrozenTransform(payload)
    cold = frozen.transform(frame)
    cold[0, 0] = 1e6
    warm = frozen.transform(frame.copy())
    assert frozen.stats['hits'] == 1 and warm[0, 0] != 1e6
    reversed_output = frozen.transform(frame.iloc[::-1])
    np.testing.assert_array_equal(reversed_output, warm[::-1])
    changed = frame.copy()
    changed.loc[0, 'a'] += 1
    frozen.transform(changed)
    assert frozen.stats['misses'] == 3
    assert frozen.stats['encoding_seconds'] > 0 and frozen.stats['hash_seconds'] > 0
    assert frozen.transform(frame.iloc[:0]).shape == (0, 2)


def test_cache_isolation_checkpoint_seed_and_payload_edits():
    payload, frame = frozen_fixture(cache_bytes=4096)
    pipeline = DAGPipePipeline(payload)
    first = pipeline.transform(frame)
    original_transform = pipeline._transforms['cpu']
    payload['reference'][0][0] += 2
    updated = pipeline.transform(frame)
    assert pipeline._transforms['cpu'] is not original_transform
    assert not np.array_equal(updated, first)
    other_payload = copy.deepcopy(payload)
    other_payload['seed'] = 43
    other = DAGPipePipeline(other_payload)
    np.testing.assert_array_equal(other.transform(frame), updated)
    assert other._transforms['cpu'] is not pipeline._transforms['cpu']
    assert other.performance_stats()['cpu']['misses'] == 1


def test_resource_guard_throttles_and_forces_boundaries():
    from unittest.mock import Mock
    check = Mock()
    with patch('baselines.DAGPipe.performance.time.monotonic', side_effect=[0., .01, .02, .12, .13]):
        guard = ResourceGuard(check, .1)
        for _ in range(4):
            guard()
        guard(force=True)
    assert check.call_count == 3


def test_cpu_budget_checks_do_not_initialize_cuda(monkeypatch):
    from baselines.common.budget import Budget, _gpu_peak_memory_mb
    from unittest.mock import Mock
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    allocate = Mock(side_effect=AssertionError('unexpected lazy CUDA context'))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', allocate)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', allocate)
    Budget().start()
    assert _gpu_peak_memory_mb() == 0
    allocate.assert_not_called()
