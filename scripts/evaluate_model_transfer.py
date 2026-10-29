"""Evaluate fixed searched pipelines across downstream models; never search.

Each source is resolved once using source_model, independently of target model.
Run --dry_run first to inspect the full matrix and validate source artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def split_names(text):
    return list(dict.fromkeys(x.strip() for x in text.split(',') if x.strip()))


def model_catalog():
    return {
        path.parent.name: yaml.safe_load(path.read_text(encoding='utf-8'))
        for path in sorted((ROOT / 'dppbench/tasks').glob('*/model.yaml'))
    }


def build_plan(args):
    catalog = model_catalog()
    tasks = split_names(args.data_names) if args.data_names else list(catalog)
    plan = []
    for task in tasks:
        if task not in catalog:
            raise ValueError(f'Unknown task: {task}')
        config = catalog[task]
        available = list(config.get('model_options') or {config['model']: {}})
        models = available if args.models == 'all' else split_names(args.models)
        if not models or set(models) - set(available):
            raise ValueError(f'{task}: requested {models}; configured models: {available}')
        source_model = args.source_model or config['model']
        for baseline in split_names(args.baselines):
            source = Path(args.pipeline_template.format(
                source_root=str(Path(args.source_root).resolve()), task=task,
                baseline=baseline, seed=args.seed, source_model=source_model,
            )).resolve()
            raw = source.read_bytes()  # Missing sources fail before any training.
            payload = yaml.safe_load(raw)
            if not isinstance(payload, dict) or not any(k in payload for k in ('dag', 'pipeline', 'format')):
                raise ValueError(f'Not a saved pipeline: {source}')
            for model in models:
                output = Path(args.output_dir).resolve() / f'seed{args.seed}' / baseline / f'{task}_{model}'
                if source.parent == output:
                    raise ValueError('Output would overwrite the source run; use a separate output_dir')
                plan.append(dict(task=task, baseline=baseline, model=model,
                                 source_model=source_model, seed=args.seed,
                                 pipeline_source=str(source),
                                 source_sha256=hashlib.sha256(raw).hexdigest(),
                                 run_output=str(output)))
    if not plan:
        raise ValueError('Empty evaluation matrix')
    return plan


def command_for(job, args):
    # Deliberately no search or HPO options. Each child has its own model state.
    command = [sys.executable, str(ROOT / 'scripts/evaluate_baseline.py'),
               '--baseline', job['baseline'], '--data_names', job['task'],
               '--model', job['model'], '--seed', str(job['seed']),
               '--resume_pipeline', job['pipeline_source'], '--skip_build',
               '--output_dir', str(Path(args.output_dir).resolve() / f"seed{job['seed']}"),
               '--output_csv', str(Path(job['run_output']) / 'result.csv'),
               '--gpu_id', str(args.gpu_id)]
    if args.data_dir:
        command += ['--data_dir', args.data_dir]
    return command


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(rows):
    """Within-task pairwise ordering agreement against the source model.

    Only matched successful baselines count; failures are never zero scores.
    No averaging AUC and RMSE across tasks, or comparisons across seed/sources.
    """
    groups = {}
    for row in rows:
        if row.get('error'):
            continue
        try:
            value = float(row['std_test_metric'])
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        key = (row['task'], row['seed'], row['source_model'], row['std_test_metric_name'])
        groups.setdefault(key, {}).setdefault(row['model'], {})[row['baseline']] = value
    result = []
    for (task, seed, reference, metric), models in groups.items():
        ref = models.get(reference, {})
        for model, scores in models.items():
            if model == reference:
                continue
            common = sorted(ref.keys() & scores.keys())
            agree = total = ties = 0
            for i, a in enumerate(common):
                for b in common[i + 1:]:
                    left, right = ref[a] - ref[b], scores[a] - scores[b]
                    if left == 0 or right == 0:
                        ties += 1
                        continue
                    total += 1
                    agree += (left > 0) == (right > 0)
            lower = metric in {'rmse', 'mse', 'mae', 'logloss'}
            ordering = sorted(scores, key=lambda b: scores[b], reverse=not lower)
            result.append(dict(task=task, seed=seed, reference_model=reference,
                               model=model, metric=metric, matched_baselines=len(common),
                               compared_pairs=total, tied_pairs=ties,
                               ordering_agreement=agree / total if total else None,
                               baseline_order=' > '.join(ordering)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list_models', action='store_true')
    parser.add_argument('--source_root')
    parser.add_argument('--pipeline_template', default='{source_root}/{baseline}/{task}_{source_model}/best_pipeline.yaml',
                        help='Supports source_root, baseline, task, source_model, seed; never target model.')
    parser.add_argument('--source_model', help='Search model; default: each task model.yaml default')
    parser.add_argument('--baselines', help='Comma-separated canonical baseline names')
    parser.add_argument('--data_names', help='Comma-separated tasks; default: all configured tasks')
    parser.add_argument('--models', default='all', help='all task-configured models, or comma-separated names')
    parser.add_argument('--seed', type=int, default=42, help='Original search seed, also used for downstream training')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir', default=str(ROOT / 'outputs/model_transfer'))
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args(argv)
    if args.list_models:
        for task, config in model_catalog().items():
            print(f"{task}: default={config['model']}; models={', '.join(config.get('model_options', {}))}")
        return 0
    if not args.source_root or not args.baselines:
        parser.error('--source_root and --baselines are required')
    plan = build_plan(args)
    root = Path(args.output_dir).resolve() / f'seed{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'plan.json'
    if manifest.exists():
        raise FileExistsError(f'Existing experiment: {manifest}; choose a new output_dir')
    manifest.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        # A preflight should not prevent the actual invocation from running.
        manifest.unlink()
        return 0
    rows = []
    for job in plan:
        output = Path(job['run_output'])
        output.mkdir(parents=True, exist_ok=True)
        row = dict(job)
        try:
            if hashlib.sha256(Path(job['pipeline_source']).read_bytes()).hexdigest() != job['source_sha256']:
                raise ValueError('Source pipeline changed after planning')
            print(f"Evaluating {job['task']} / {job['baseline']} / {job['model']}", flush=True)
            with (output / 'stdout.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command_for(job, args), stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f'Evaluation exited {completed.returncode}; see {output / "stdout.log"}')
            with (output / 'result.csv').open(encoding='utf-8', newline='') as stream:
                child_rows = list(csv.DictReader(stream))
            if len(child_rows) != 1:
                raise ValueError('Expected one evaluation row')
            row.update(child_rows[0])
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        rows.append(row)
        write_csv(root / 'results.csv', rows)
        write_csv(root / 'comparisons.csv', comparison_rows(rows))
    print(f'Results: {root / "results.csv"}')
    return int(any(row.get('error') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
