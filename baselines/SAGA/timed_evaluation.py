"""Spawned evaluation: a stalled native imputer can actually be terminated."""
import contextlib
from dataclasses import asdict
import json
import math
import multiprocessing
import os
import random
import signal
import shutil
import tempfile
from pathlib import Path
import time

from baselines.common.evaluator import EvaluationResult


def _rng_state(device):
    import numpy as np
    import torch
    state = np.random.get_state()
    result = {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
              'torch': torch.get_rng_state().tolist()}
    if str(device).startswith('cuda'):
        result['cuda'] = torch.cuda.get_rng_state(device).tolist()
    return result


def _restore_rng_state(state, device):
    import numpy as np
    import torch
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(state['python']))
    arr = state['numpy']
    np.random.set_state((arr[0], np.array(arr[1], dtype=np.uint32), *arr[2:]))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    if 'cuda' in state:
        torch.cuda.set_rng_state(torch.tensor(state['cuda'], dtype=torch.uint8), device)


def _worker(request, yaml_text, result_path, log_path):
    if os.name == 'posix':
        os.setsid()
        Path(log_path).with_name('process_group.json').write_text(json.dumps({'pid': os.getpid()}))
    from baselines.common.pipeline import Pipeline
    from baselines.common.search_evaluator import SearchPipelineEvaluator
    from baselines.common.budget import Budget, budget_scope
    evaluator = None
    # A killed worker leaves any temporary artifacts inside its trial directory.
    # Windows native ML libraries may not support non-ASCII temp paths. Keep
    # their OS temp directory and record the owned directory for cleanup.
    if os.name == 'posix':
        tempfile.tempdir = str(Path(log_path).resolve().parent)
    with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                import numpy as np
                import torch
                random.seed(request['seed'])
                np.random.seed(request['seed'])
                torch.manual_seed(request['seed'])
                if str(request['device']).startswith('cuda'):
                    torch.cuda.set_device(request['device'])
                    torch.cuda.manual_seed_all(request['seed'])
                _restore_rng_state(request.pop('rng_state'), request['device'])
                fast_mode = request.pop('fast_mode')
                overrides = request.pop('fast_overrides')
                model_config = request.pop('model_config')
                limits = request.pop('limits')
                worker_budget = Budget(**limits)
                worker_budget.start()
                with budget_scope(worker_budget):
                    evaluator = SearchPipelineEvaluator(**request, budget=worker_budget, model_config=model_config)
                    Path(log_path).with_name('worker_temp.json').write_text(json.dumps({
                        'path': evaluator._executor._work_dir}), encoding='utf-8')
                    evaluator.set_evaluate_std_test(False)
                    evaluator.set_fast_mode(fast_mode, overrides)
                    result = evaluator.evaluate(Pipeline.from_yaml(yaml_text))
                    result.metrics['_candidate_budget'] = worker_budget.snapshot()
            except Exception as exc:
                result = EvaluationResult(fitness=-math.inf, success=False, metrics={},
                                          error=f'{type(exc).__name__}: {exc}')
            finally:
                if evaluator is not None:
                    evaluator._executor.cleanup()
            result.metrics['_rng_state'] = _rng_state(request['device'])
            # Small durable result avoids multiprocessing pipe buffer deadlocks.
            Path(result_path).write_text(json.dumps(asdict(result)), encoding='utf-8')


def evaluate_with_timeout(evaluator, pipeline, seconds, directory, isolate=False):
    if seconds < 0:
        raise ValueError('candidate_timeout_s must be nonnegative')
    if seconds == 0 and not isolate:
        return evaluator.evaluate(pipeline)
    # Only needed for opt-in isolation. Enumerate/account for descendants and
    # retain Process identities (including creation time) against PID reuse.
    import psutil
    key = f'{evaluator.scoring_method}:{pipeline.hash()}'
    if key in evaluator._cache:
        return evaluator._cache[key]
    budget = evaluator.budget
    if budget:
        budget.consume_evaluation()
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output/'result.json'
    result_path.unlink(missing_ok=True)
    group_path = output/'process_group.json'
    group_path.unlink(missing_ok=True)
    (output/'worker_temp.json').unlink(missing_ok=True)
    request = dict(task_dir=evaluator.task_dir, data_name=evaluator.data_name,
                   data_dir=evaluator.data_dir, metric_key=evaluator.metric_key,
                   verbose=True, small_n=evaluator.small_n, seed=evaluator.seed,
                   device=evaluator.device, model_name=evaluator.model_name,
                   fast_mode=evaluator._executor._fast_mode,
                   fast_overrides=evaluator._executor._fast_overrides)
    request['model_config'] = getattr(evaluator._executor, 'model_config', None)
    request['rng_state'] = _rng_state(evaluator.device)
    request['limits'] = {name: getattr(budget, name) for name in (
        'max_pipeline_length', 'max_memory_mb', 'max_gpu_memory_mb',
        'max_output_rows', 'max_output_columns')} if budget else {}
    if seconds:
        request['limits']['max_seconds'] = seconds
    process = multiprocessing.get_context('spawn').Process(
        target=_worker, args=(request, pipeline.to_yaml(), str(result_path), str(output/'worker.log')))
    started = time.monotonic()
    process.start()
    descendants = {}
    try:
        deadline = started+seconds if seconds else float('inf')
        while process.is_alive() and time.monotonic() < deadline:
            process.join(timeout=min(.2, max(0, deadline-time.monotonic())))
            try:
                root = psutil.Process(process.pid)
                for child in root.children(recursive=True):
                    descendants[(child.pid, child.create_time())] = child
            except psutil.Error:
                root = None
            if budget:
                # Linux server: account for worker RSS while it is stuck inside
                # a native operation, where cooperative checks cannot execute.
                from baselines.common.budget import _process_peak_memory_mb
                rss = _process_peak_memory_mb()
                for child in ([root] if root else [])+list(descendants.values()):
                    try:
                        rss += child.memory_info().rss/1024**2
                    except psutil.Error:
                        pass
                budget.peak_memory_mb = max(budget.peak_memory_mb, rss)
                budget.check_resources()
        if process.is_alive():
            result = EvaluationResult(fitness=-math.inf, success=False, metrics={},
                                      error=f'CandidateTimeout: exceeded {seconds}s (including spawn/loading)')
        elif result_path.exists() and process.exitcode == 0:
            try:
                result = EvaluationResult(**json.loads(result_path.read_text(encoding='utf-8')))
            except (ValueError, TypeError) as exc:
                result = EvaluationResult(fitness=-math.inf, success=False, metrics={},
                                          error=f'CandidateResultError: {exc}; see worker.log')
            if '_rng_state' in result.metrics:
                _restore_rng_state(result.metrics.pop('_rng_state'), evaluator.device)
            if budget:
                stats = result.metrics.get('_candidate_budget', {})
                budget.peak_gpu_memory_mb = max(budget.peak_gpu_memory_mb, stats.get('peak_gpu_memory_mb', 0))
                budget.check_resources()
        else:
            result = EvaluationResult(fitness=-math.inf, success=False, metrics={},
                                      error=f'CandidateProcessError: exit={process.exitcode}; see worker.log')
    finally:
        # Kill the owned POSIX process group even if its leader already exited.
        if os.name == 'posix' and group_path.exists():
            if json.loads(group_path.read_text())['pid'] == process.pid:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in descendants.values():
            try:
                child.terminate()
            except psutil.Error:
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
        if os.name == 'posix' and group_path.exists():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, surviving = psutil.wait_procs(list(descendants.values()), timeout=2)
        for child in surviving:
            try:
                child.kill()
            except psutil.Error:
                pass
        process.join()
        process.close()
        temporary_path = output/'worker_temp.json'
        if os.name != 'posix' and temporary_path.exists():
            owned = Path(json.loads(temporary_path.read_text(encoding='utf-8'))['path']).resolve()
            if owned.parent == Path(tempfile.gettempdir()).resolve() and owned.name.startswith('preproc_opt_'):
                shutil.rmtree(owned, ignore_errors=True)
    result.duration_seconds = time.monotonic()-started
    evaluator._cache[key] = result
    return result
