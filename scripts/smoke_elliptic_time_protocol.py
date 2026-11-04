"""Bounded SAGA acceptance, seed42. Requires existing data; never downloads it."""
import argparse
import json
import contextlib
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Directory containing the three Elliptic CSVs')
    parser.add_argument('--output-dir', required=True, help='New diagnostic directory')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    data = Path(args.data_dir)
    required = ['elliptic_txs_features.csv', 'elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv']
    missing = [str(data / name) for name in required if not (data / name).is_file()]
    if missing:
        raise SystemExit('BLOCKED: existing data required; no download attempted: ' + ', '.join(missing))
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit('Use a new output directory to preserve previous experiments')
    from baselines.SAGA.saga import SAGA
    from baselines.common.budget import Budget, budget_scope
    runner = SAGA(task_dir=str(Path(__file__).resolve().parents[1] / 'dppbench/tasks/elliptic_bitcoin'),
        data_name='elliptic_bitcoin', data_dir=str(data), seed=42, population_size=4,
        n_generations=1, top_k=1, n_physical_trials=1, small_n=3000, fast_train=True,
        max_pipeline_len=10, device=args.device, output_dir=str(output), skip_final_eval=True)
    budget = Budget(max_pipeline_length=10, max_evaluations=16, max_seconds=300)
    output.mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, original, log):
            self.original, self.log = original, log
        def write(self, text):
            self.original.write(text); self.log.write(text); self.log.flush()
        def flush(self):
            self.original.flush(); self.log.flush()
    with (output / 'acceptance.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            with budget_scope(budget):
                result = runner.run()
    (output / 'acceptance_result.json').write_text(json.dumps(result, default=str, indent=2), encoding='utf-8')
    text = (output / 'acceptance.log').read_text(encoding='utf-8')
    failures = [line for line in text.splitlines() if 'FAILED' in line or 'Traceback' in line]
    passed = (result.get('is_legal') and math.isfinite(result.get('best_fitness', -math.inf))
              and not result.get('eval_error') and not failures)
    (output / 'acceptance_status.json').write_text(json.dumps({'passed': bool(passed),
        'failure_lines': failures, 'scope': 'seed42 bounded search, no final HPO/test'}, indent=2), encoding='utf-8')
    if not passed:
        raise SystemExit('Acceptance NOT passed; inspect acceptance.log and acceptance_status.json')
    print('Acceptance passed: finite fitness, legal pipeline, no reported candidate failures.')


if __name__ == '__main__':
    main()
