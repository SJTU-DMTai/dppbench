#!/usr/bin/env python3
"""Berka DiffPrep post-search pipeline ablations (no DiffPrep retraining)."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / 'baselines').exists() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This template is reconstructed from the user-provided OLD seed-43 log.
# Critically: it is *not* a byte-for-byte recovered historical YAML.
OLD_TEMPLATE = '''dag:
  sources:
  - id: s0
    table: main
  - id: s1
    table: order
  ops:
  - id: o1
    op: JoinTable
    prev: [s0, s1]
    params:
      method: agg
      key_col: account_id
      how: left
  - id: o2
    op: HandleMV
    prev: [o1]
    params:
      method: median
      action: impute
      cols: [account_id, amount]
  - id: o3
    op: LabelEncode
    prev: [o2]
  - id: o4
    op: ExtractDateTimeFeature
    prev: [o3]
    params:
      cols: [account_id, duration, payments]
  - id: o5
    op: FrequencyEncode
    prev: [o4]
    params:
      cols: [amount]
  train:
    prev: [o5]
'''

CASES = {
    'order_full': ('order', True, True),
    'order_no_date': ('order', False, True),
    'order_no_freq': ('order', True, False),
    'order_no_date_no_freq': ('order', False, False),
    'account_full': ('account', True, True),
    'account_no_date': ('account', False, True),
    'account_no_freq': ('account', True, False),
    'account_no_date_no_freq': ('account', False, False),
}


def make_pipeline(case, source_yaml=None):
    from baselines.common.pipeline import Pipeline
    import copy
    join, use_date, use_freq = CASES[case]
    origin = Path(source_yaml).read_text(encoding='utf-8') if source_yaml else OLD_TEMPLATE
    p = Pipeline.from_yaml(origin)
    p = p.copy()
    joins = [s for s in p.steps if s.op == 'JoinTable']
    if len(joins) != 1:
        raise ValueError('Expected exactly one JoinTable in source YAML')
    jp = joins[0]
    # Only change the join, keep the other parameters exactly as in source.
    jp.params = copy.deepcopy(jp.params)
    jp.params['method'] = 'agg' if join == 'order' else 'key'
    jp.params['key_col'] = 'account_id'
    jp.params['how'] = 'left'
    jp.params.pop('aux_df', None)
    jp.params.pop('prefix', None)
    # prefix/max_cols are ignored for this historical order pipeline,
    # but avoid carrying restrictions across table types.
    if join == 'order':
        jp.params.pop('max_cols', None)
    if p.sources is None:
        raise ValueError('Expected DAG sources in source YAML')
    source_ids = [x for x in p.sources if x.get('id') != 's0']
    if len(source_ids) != 1:
        raise ValueError('Expected one auxiliary source')
    source_ids[0]['table'] = join
    if not use_date:
        p.steps = [s for s in p.steps if s.op != 'ExtractDateTimeFeature']
    if not use_freq:
        p.steps = [s for s in p.steps if s.op != 'FrequencyEncode']
    # Reconstruct linear edges after removing an intermediate DAG node.
    previous = 'o1'
    for i, step in enumerate(p.steps):
        if step.op == 'JoinTable':
            step.prev = ['s0', source_ids[0]['id']]
        else:
            step.prev = [previous]
        previous = step.id
    p.train_prev = [previous]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', required=True, choices=CASES)
    ap.add_argument('--task-dir', required=True, help='Directory containing Berka model.yaml and prepare.yaml')
    ap.add_argument('--data-dir', default=None, help='Optional Berka raw/frozen data directory')
    ap.add_argument('--source-yaml', default=None, help='Optional historical pipeline YAML; absent uses log-reconstructed template')
    ap.add_argument('--out', required=True, help='Directory for ablation results')
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f'{args.case}.json'
    row = {'case': args.case, 'seed': args.seed, 'success': False, 'fitness': None, 'error': None}
    try:
        from baselines.common.evaluator import PipelineEvaluator
        p = make_pipeline(args.case, args.source_yaml)
        (out / f'{args.case}.yaml').write_text(p.to_yaml(), encoding='utf-8')
        # Real validation evaluation only; no std_test score is used or logged here.
        evaluator = PipelineEvaluator(
            task_dir=str(Path(args.task_dir).resolve()),
            data_name='berka', data_dir=args.data_dir,
            model_name='LightGBM', scoring_method='full_train',
            seed=args.seed, verbose=True,
        )
        ev = evaluator.evaluate(p, scoring_method='full_train')
        row.update(success=bool(ev.success), fitness=float(ev.fitness) if ev.success else None,
                   metrics=ev.metrics, error=ev.error, duration_seconds=ev.duration_seconds)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    result_file.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ABLATION_RESULT=' + json.dumps(row, ensure_ascii=False))
    return 0 if row['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
