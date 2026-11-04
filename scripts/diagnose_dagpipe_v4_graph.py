"""Replay a fixed Elliptic v4 checkpoint and compare validation-only GraphSAGE arms."""
import argparse
import contextlib
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines.DAGPipe.pipeline import DAGPipePipeline
from baselines.DAGPipe.runtime import FrozenTransform
from dppbench.operators.preprocessing.scale_feature import ScaleFeature


def distribution(x, columns):
    absolute = np.abs(x)
    q = np.quantile(absolute, [.5, .95, .99, .999], axis=0)
    peaks = absolute.max(axis=0)
    return {name: dict(p50=float(q[0, i]), p95=float(q[1, i]), p99=float(q[2, i]),
                       p999=float(q[3, i]), abs_max=float(peaks[i]))
            for i, name in enumerate(columns)}


def scale_from_train(x, mask, columns, method):
    if not mask.any():
        raise ValueError('Empty supervised training mask')
    frame = pd.DataFrame(x, columns=columns)
    scaler = ScaleFeature(cols=columns, method=method)
    # Fit at float64 precision on the exact values consumed by GraphSAGE.
    scaler.transform(frame.loc[mask].astype(np.float64))
    out = scaler.transform(frame.astype(np.float64)).to_numpy(dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f'{method} produced nonfinite GraphSAGE input')
    return out, scaler.params_


def trace_rows(frozen, encoded, columns):
    events = []

    def observe(stage, tensor, layer=None, node=None, **details):
        array = tensor.detach().cpu().numpy()
        row, col = np.unravel_index(np.abs(array).argmax(), array.shape)
        event = dict(stage=stage, layer=layer, node=node, probe_row=int(row),
                     column=columns[col], value=float(array[row, col]))
        if node is not None:
            op = frozen.model.op_layers[layer][node]
            event['operator'] = op.name
            if op.frozen_stats and col in op.numeric:
                event['fitted'] = {k: v[op.numeric.index(col)] for k, v in op.frozen_stats.items()}
        for key in ('source_node', 'source_layer', 'weight'):
            if key in details:
                event[key] = float(details[key])
        if 'routing' in details:
            event['routing'] = float(details['routing'][0, col])
        events.append(event)

    frozen.model.forward_observer = observe
    try:
        with torch.no_grad():
            frozen.model(encoded, reference_rows=0)
    finally:
        del frozen.model.forward_observer
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-dir', help='DPPBench data root containing elliptic_bitcoin/data')
    parser.add_argument('--hpo-report', type=Path, help='Original final_hpo.json; reuses best_params')
    parser.add_argument('--expect-hash')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--reference-only', action='store_true', help='Trace checkpoint reference rows without a dataset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.reference_only and not (args.data_dir and args.hpo_report):
        parser.error('Graph comparison requires --data-dir and --hpo-report')
    pipeline = DAGPipePipeline.from_yaml(args.checkpoint.read_text(encoding='utf-8'))
    p = pipeline.payload
    if p['format'] != 'dppbench.DAGPipe.reference.v4' or p['task'] != 'elliptic_bitcoin':
        parser.error('Expected an Elliptic DAGPipe v4 checkpoint')
    digest = pipeline.hash()
    if args.expect_hash and digest != args.expect_hash:
        parser.error(f'Checkpoint hash mismatch: {digest}')
    columns = [s['column'] for s in p['encoder']]
    report = dict(pipeline_hash=digest, fit_protocol=p['fit_protocol'], training_rows=p['training_rows'],
                  reference_rows=len(p['reference']), seed=p['seed'], torch=torch.__version__)
    with args.output.open('x', encoding='utf-8') as output:
        if args.reference_only:
            frozen = FrozenTransform(p, args.device)
            encoded = torch.tensor(p['reference'], dtype=torch.float64, device=args.device)
            report.update(scope='checkpoint_reference_rows_only', trace=trace_rows(frozen, encoded, columns))
        else:
            run_graph(args, pipeline, columns, report)
        json.dump(report, output, indent=2, allow_nan=False)


def run_graph(args, pipeline, columns, report, methods=None):
    from baselines.DAGPipe.evaluator import DAGPipeExecutor
    from dppbench.models import GraphSAGE, train_graph

    params = (json.loads(args.hpo_report.read_text(encoding='utf-8'))['best_params']
              if args.hpo_report else {})
    if not isinstance(params, dict):
        raise ValueError('Expected selected best_params in original HPO report')
    task = 'elliptic_bitcoin'
    root = Path(__file__).resolve().parents[1]
    executor = DAGPipeExecutor(str(root / 'dppbench/tasks' / task), task,
                               data_dir=str(Path(args.data_dir) / task / 'data'),
                               model_name='GraphSAGE', device=args.device, model_config=params,
                               seed=pipeline.payload['seed'])
    probe = {}
    original = pipeline.transform

    def transform(frame, device):
        values = original(frame, device)
        # Trace rows responsible for the largest output in any feature column.
        indices = np.unique(np.abs(values).argmax(axis=0))
        frozen = pipeline._transforms[device]
        encoded = torch.as_tensor(frozen.encoder.transform(frame.iloc[indices]), dtype=torch.float64, device=device)
        probe.update(graph_row_positions=indices.tolist(), trace=trace_rows(frozen, encoded, columns))
        return values

    pipeline.transform = transform
    try:
        executor.set_evaluate_std_test(False)
        executor.pipeline = pipeline
        data = executor._make_configured_data_instance()
        data.load_data()
        data.run_pre_process(executor.original_yaml)
        if [c for c in data.train_df if c.startswith('feat_')] != columns:
            raise ValueError('Graph feature order differs from checkpoint schema')
        graph = data.build_graph()
        cfg = executor.get_model_config()
    finally:
        pipeline.transform = original
        executor.cleanup()
    if np.any(graph.get('std_test_mask', False)):
        raise ValueError('Unexpected standard-test nodes in validation graph')
    x, mask, val = graph['x'], graph['train_mask'], graph['val_mask']
    if np.any(mask & val) or mask.sum() != pipeline.payload['training_rows']:
        raise ValueError('Graph training mask differs from checkpoint training scope')
    report.update(scope='validation_graph', graph_rows=len(x), train_nodes=int(mask.sum()),
                  val_nodes=int(val.sum()), model_config=cfg, peak_row_probe=probe, arms={})
    ctor = {k: v for k, v in cfg['model_params'].items() if k not in ('task', 'epochs', 'lr', 'weight_decay', 'seed')}
    seed = int(cfg['model_params'].get('seed', 42))
    if methods is None:
        methods = ['raw', 'standard'] + (['robust'] if args.robust else [])
    for arm in methods:
        features, fitted = (x, None) if arm == 'raw' else scale_from_train(x, mask, columns, arm)
        result = dict(all_nodes=distribution(features, columns),
                      train=distribution(features[mask], columns), val=distribution(features[val], columns),
                      abs_max=float(np.abs(features).max()), fit_params=fitted)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = GraphSAGE(in_dim=x.shape[1], **ctor)
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                metrics = train_graph(model, dict(graph, x=features), cfg['model_params'], cfg['train'],
                                      device=args.device, evaluate_test=False)
            result['val_auc'] = float(metrics['val_auc'])
        except (ValueError, RuntimeError) as exc:
            result['error'] = str(exc)
        finally:
            print(f'[{arm}]\n{stream.getvalue()}', end='', flush=True)
        result['curve'] = [dict(epoch=int(epoch), loss=float(loss), val_auc=float(auc))
                           for epoch, loss, auc in re.findall(r'epoch (\d+)/\d+ loss=([\deE.+-]+) val_auc=([\deE.+-]+)', stream.getvalue())]
        report['arms'][arm] = result


if __name__ == '__main__':
    main()
