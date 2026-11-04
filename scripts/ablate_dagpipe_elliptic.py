"""Run paired v4 Elliptic scaler or reference-size ablations."""

import argparse
import json
from pathlib import Path

import yaml

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-rows', type=int,
                        help='Compare 16 rows against this larger reference, keeping the full candidate layout')
    parser.add_argument('--hpo-report', type=Path, help='Optional shared GraphSAGE parameters from an existing final_hpo.json')
    args = parser.parse_args()
    if args.reference_rows is not None and args.reference_rows <= 16:
        parser.error('--reference-rows must be greater than 16')

    root = Path(__file__).resolve().parents[1]
    task = 'elliptic_bitcoin'
    data_dir = args.data_dir / task / 'data'
    if not data_dir.is_dir():
        parser.error(f'Missing dataset directory: {data_dir}')

    from baselines.DAGPipe.dagpipe import DAGPipe
    from scripts.diagnose_dagpipe_v4_graph import run_graph

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = root / 'baselines/DAGPipe/config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    summary = {}

    if args.reference_rows is None:
        arms = [('control', {}), ('no_second_robust', {'ablate_elliptic_second_robust_scaler': True})]
    else:
        arms = [('ref16', {}), (f'ref{args.reference_rows}', {'elliptic_reference_rows': args.reference_rows})]
    config.update(ablate_elliptic_second_robust_scaler=False, elliptic_reference_rows=16)
    for arm, overrides in arms:
        print(f'[{arm}] seed={args.seed}', flush=True)
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        arm_config = arm_dir / 'config.yaml'
        arm_config.write_text(yaml.safe_dump(dict(config, **overrides)), encoding='utf-8')
        runner = DAGPipe(
            task_dir=str(root / 'dppbench/tasks' / task), data_name=task,
            data_dir=str(data_dir), seed=args.seed, output_dir=arm_dir,
            device=args.device, max_pipeline_len=10, model_name='GraphSAGE',
            config_path=arm_config,
        )
        runner.run()
        pipeline = runner.get_pipeline()
        columns = [spec['column'] for spec in pipeline.payload['encoder']]
        report = dict(pipeline_hash=pipeline.hash(), layers=pipeline.payload['layers'])
        run_graph(args, pipeline, columns, report, methods=['raw'])
        (arm_dir / 'validation.json').write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding='utf-8',
        )
        metrics = report['arms']['raw']
        if 'error' in metrics:
            raise RuntimeError(f"{arm} GraphSAGE evaluation failed: {metrics['error']}")
        summary[arm] = dict(
            layers=report['layers'], reference_rows=len(pipeline.payload['reference']),
            val_auc=metrics['val_auc'],
            abs_max=metrics['abs_max'],
            max_feature_p999=max(stats['p999'] for stats in metrics['all_nodes'].values()),
        )

    output = json.dumps(summary, indent=2, allow_nan=False)
    (args.output_dir / 'summary.json').write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
