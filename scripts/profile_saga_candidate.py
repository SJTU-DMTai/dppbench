"""Compare one known-fast pipeline inline vs spawned, without a search timeout.

This is an opt-in overhead diagnostic, not a SAGA search or final test run.
Choose a previously saved, inexpensive pipeline. --timeout_s explicitly changes
the spawned candidate protocol; its default is zero (no cutoff).
"""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--task', default='fraud_detection')
    parser.add_argument('--model', default='LightGBM')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--small_n', type=int, default=3000)
    parser.add_argument('--timeout_s', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    if args.timeout_s < 0 or args.small_n < 0:
        parser.error('timeout_s and small_n must be nonnegative')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from baselines.common.pipeline import Pipeline
    from baselines.common.search_evaluator import SearchPipelineEvaluator
    from baselines.SAGA.timed_evaluation import evaluate_with_timeout, _rng_state, _restore_rng_state

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    pipeline = Pipeline.from_yaml(Path(args.pipeline).read_text(encoding='utf-8'))
    kwargs = dict(task_dir=str(ROOT/'dppbench/tasks'/args.task), data_name=args.task,
                  data_dir=args.data_dir, model_name=args.model, device=args.device,
                  small_n=args.small_n, seed=args.seed, verbose=True)
    inline, isolated = SearchPipelineEvaluator(**kwargs), SearchPipelineEvaluator(**kwargs)
    try:
        for evaluator in (inline, isolated):
            evaluator.set_evaluate_std_test(False)
            evaluator.set_fast_mode(True)
        state = _rng_state(args.device)
        first = inline.evaluate(pipeline)
        _restore_rng_state(state, args.device)
        second = evaluate_with_timeout(isolated, pipeline, args.timeout_s, output/'worker', isolate=True)
        summary = {'diagnostic_only': True, 'pipeline_hash': pipeline.hash(),
                   'timeout_s': args.timeout_s, 'seed': args.seed, 'small_n': args.small_n,
                   'inline': {'success': first.success, 'fitness': first.fitness if first.success else None,
                              'seconds': first.duration_seconds, 'error': first.error},
                   'isolated': {'success': second.success, 'fitness': second.fitness if second.success else None,
                                'seconds_including_spawn': second.duration_seconds, 'error': second.error}}
        (output/'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        if not first.success or not second.success:
            raise RuntimeError(f'Candidate diagnostic failed; inspect {output}/comparison.json')
    finally:
        inline._executor.cleanup(); isolated._executor.cleanup()


if __name__ == '__main__':
    main()
