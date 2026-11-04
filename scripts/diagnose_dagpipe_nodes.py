"""Trace actual DAGPipe checkpoint replay on training references and validation features."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines.DAGPipe.pipeline import DAGPipePipeline
from baselines.DAGPipe.runtime import FrozenTransform, reference_forward


THRESHOLDS = (10_000, 1_000_000, 100_000_000)
V2_PROTOCOLS = {'relative-floor': None, 'absolute-epsilon': 'absolute_epsilon_v2'}
STATS = ('min', 'max', 'p50', 'p95', 'p99', 'p99_9', 'abs_max', 'nan_count', 'inf_count')


def distribution(values):
    finite = values[np.isfinite(values)]
    result = dict.fromkeys(STATS)
    result.update(nan_count=int(np.isnan(values).sum()), inf_count=int(np.isinf(values).sum()))
    if finite.size:
        result.update(min=float(finite.min()), max=float(finite.max()),
                      abs_max=float(np.abs(finite).max()))
        result.update(zip(('p50', 'p95', 'p99', 'p99_9'),
                          map(float, np.percentile(finite, [50, 95, 99, 99.9]))))
    return result


def trace_checkpoint(payload, query_frame=None, *, v2_protocol=None, device="cpu", compare_public_scalers=False):
    checkpoint_format = payload.get('format')
    if checkpoint_format == 'dppbench.DAGPipe.reference.v2':
        if v2_protocol not in V2_PROTOCOLS:
            raise ValueError('V2 does not identify its historical scaling rule. Supply --v2-scale-protocol from the source used to train it.')
        if payload.get('operator_space', 'current') != 'current' or payload.get('scale_protocol') is not None:
            raise ValueError('Unrecognized v2 execution metadata')
    elif v2_protocol is not None:
        raise ValueError('--v2-scale-protocol applies only to v2 checkpoints')
    DAGPipePipeline(payload)
    # Instantiate exactly the supported replay path with an explicit v2 choice.
    replay_payload = dict(payload)
    if checkpoint_format == 'dppbench.DAGPipe.reference.v2':
        replay_payload['scale_protocol'] = V2_PROTOCOLS[v2_protocol]
    frozen = FrozenTransform(replay_payload, device=device)
    reference = frozen.reference
    query = reference if query_frame is None else reference.new_tensor(frozen.encoder.transform(query_frame))
    if not len(query):
        raise ValueError('Validation feature file has no rows')
    columns = [spec['column'] for spec in frozen.encoder.state]
    n_ref = len(reference)
    comparison = None
    if compare_public_scalers:
        from scripts.dagpipe_scaler_comparison import ScalerComparison
        comparison = ScalerComparison(payload, n_ref)
    events, scalers, first = [], [], {}
    robust_peaks, routed_inputs, reference_stats = {}, {}, {}
    current_stats, routed_ranges = {}, {}
    offset, cursor = 0, 0
    reference_stats_changes = 0
    output_abs_max = 0.0

    def observe(stage, value, layer=None, node=None, routing=None, **details):
        nonlocal cursor, reference_stats_changes
        if comparison is not None:
            comparison.observe(stage, value, layer=layer, node=node)
        event = {'event': cursor, 'stage': stage, 'layer': layer, 'node': node,
                 'op': payload['layers'][layer][node] if node is not None else None}
        if routing is not None:
            event['routing_weights'] = routing.detach().flatten().cpu().tolist()
        for key in ('weight', 'source_layer', 'source_node'):
            if key in details:
                item = details[key]
                event[key] = float(item.detach()) if torch.is_tensor(item) else item
        cursor += 1
        if offset == 0:
            events.append(event)
        if stage == 'scaler_statistics':
            fields = {k: v.detach().flatten().cpu().tolist() for k, v in details.items()}
            key = (layer, node)
            current_stats[key] = fields
            if offset:
                for name, values in fields.items():
                    if not np.allclose(values, reference_stats[key][name], rtol=1e-9, atol=1e-12, equal_nan=True):
                        reference_stats_changes += 1
                return
            reference_stats[key] = fields
            for i, column in enumerate(columns):
                row = {k: event[k] for k in ('event', 'layer', 'node', 'op')}
                row.update(column=column, routing_weight=event['routing_weights'][i])
                row.update({k: v[i] if isinstance(v[i], bool) or np.isfinite(v[i]) else None
                            for k, v in fields.items()})
                scalers.append(row)
            return
        array = value.detach().cpu().numpy()
        key = (layer, node)
        if event['op'] == 'RobustScaler':
            if stage == 'after_routing':
                routed_inputs[key] = array
                block = array[n_ref:]
                valid = np.isfinite(block)
                low = np.where(valid, block, np.inf).min(axis=0)
                high = np.where(valid, block, -np.inf).max(axis=0)
                old_low, old_high = routed_ranges.get(key, (low, high))
                routed_ranges[key] = np.minimum(low, old_low), np.maximum(high, old_high)
            elif stage == 'operator_output':
                before = routed_inputs.pop(key)
                for split, start, stop in [('reference', 0, n_ref), ('query', n_ref, len(array))]:
                    block = array[start:stop]
                    finite_abs = np.where(np.isfinite(block), np.abs(block), -np.inf)
                    row, col = np.unravel_index(finite_abs.argmax(), block.shape)
                    magnitude = float(finite_abs[row, col])
                    peak_key = (layer, node, split)
                    if not np.isfinite(magnitude) or magnitude <= robust_peaks.get(peak_key, {}).get('abs_max', -1):
                        continue
                    position = int(row) + (offset if split == 'query' else 0)
                    peak = dict(layer=layer, node=node, split=split, row_position=position,
                                column=columns[col], abs_max=magnitude,
                                scaler_input=float(before[start + row, col]), scaler_output=float(block[row, col]),
                                routing_weight=event['routing_weights'][col],
                                reference_input_distribution=distribution(before[:n_ref, col]))
                    peak.update({name: values[col] for name, values in current_stats[key].items()})
                    peak['median'] = peak['center']
                    peak['iqr'] = peak['raw_scale']
                    peak['eps'] = None if payload.get('scale_protocol') == 'public_v1' else 1e-8
                    peak['eps_policy'] = ('positive IQR, no EPS floor' if peak['eps'] is None
                                          else f'{v2_protocol}, absolute epsilon 1e-8')
                    if split == 'query' and query_frame is not None:
                        record = query_frame.iloc[position]
                        if 'txId' in record:
                            peak['txId'] = str(record['txId'])
                        peak['raw_feature_value'] = str(record[columns[col]])
                    robust_peaks[peak_key] = peak
        for split, data in [('reference', array[:n_ref]), ('query', array[n_ref:])]:
            if offset and split == 'reference':
                continue
            if offset and not any(
                    ((f'{split}:{t}' not in first or event['event'] < first[f'{split}:{t}']['event'])
                     and np.any(np.abs(data) > t)) for t in THRESHOLDS):
                continue
            summary = distribution(data)
            per_column = [distribution(data[:, i]) for i in range(len(columns))]
            event[split] = {'summary': summary, 'columns': per_column}
            for threshold in THRESHOLDS:
                key = f'{split}:{threshold}'
                if (key not in first or event['event'] < first[key]['event']) and (summary['abs_max'] or 0) > threshold:
                    indices = [i for i, row in enumerate(per_column) if (row['abs_max'] or 0) > threshold]
                    first[key] = {k: event[k] for k in ('event', 'stage', 'layer', 'node', 'op')}
                    first[key]['columns'] = [columns[i] for i in indices]
                    first[key]['batch_start'] = offset

    frozen.model.forward_observer = observe
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(query), frozen.batch_size):
            cursor = 0
            batch = query[offset:offset + frozen.batch_size]
            observe('input_encoding', torch.cat([reference, batch]))
            output = reference_forward(frozen.model, batch, reference)
            output_abs_max = max(output_abs_max, float(output.abs().max()))
            outputs.append(output.cpu())
        final = events[-1]
        events.append(dict(final, event=len(events), stage='final_output'))
    output = torch.cat(outputs)
    for (layer, node, _), peak in robust_peaks.items():
        i = columns.index(peak['column'])
        low, high = routed_ranges[layer, node]
        peak['full_query_routed_input_min'] = float(low[i]) if np.isfinite(low[i]) else None
        peak['full_query_routed_input_max'] = float(high[i]) if np.isfinite(high[i]) else None
        peak['full_query_encoded_distribution'] = distribution(query[:, i].cpu().numpy())
        peak['reference_encoded_distribution'] = distribution(reference[:, i].cpu().numpy())
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite final output, which the normal replay path also rejects')
    return {'task': payload['task'], 'format': checkpoint_format,
            'operator_space': payload.get('operator_space', 'current'),
            'scale_protocol': v2_protocol or payload.get('scale_protocol'),
            'query_source': 'checkpoint_reference' if query_frame is None else 'validation_features',
            'reference_rows': n_ref, 'query_rows': len(query), 'columns': columns,
            'layers': payload['layers'], 'first_threshold_crossings': first,
            'events': events, 'scaler_statistics': scalers,
            'robust_scaler_peaks': list(robust_peaks.values()),
            'reference_statistics_changes_across_batches': reference_stats_changes,
            'batch_size': frozen.batch_size, 'event_distribution_scope': 'first query batch',
            'full_query_output_abs_max': output_abs_max,
            'public_scaler_comparison': comparison.result() if comparison is not None else [],
            'full_query_input_distribution': distribution(query.cpu().numpy()),
            'reference_input_distribution': distribution(reference.cpu().numpy())}, output


def write_reports(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'diagnosis.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    identifiers = ('event', 'stage', 'layer', 'node', 'op', 'source_layer', 'source_node', 'weight')
    with (output_dir / 'node_statistics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[*identifiers, 'split', 'column', 'routing_weight', *STATS])
        writer.writeheader()
        for event in report['events']:
            base = {k: event.get(k) for k in identifiers}
            for split in ('reference', 'query'):
                if split not in event:
                    continue
                values = event[split]
                writer.writerow(dict(base, split=split, column='', **values['summary']))
                for i, (name, stats) in enumerate(zip(report['columns'], values['columns'])):
                    weights = event.get('routing_weights')
                    writer.writerow(dict(base, split=split, column=name,
                                         routing_weight=weights[i] if weights else None, **stats))
    rows = report['scaler_statistics']
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / 'scaler_statistics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields or ['event', 'layer', 'node', 'op', 'column'])
        writer.writeheader()
        writer.writerows(rows)
    comparison = report.get('training_reference_comparison', [])
    public_rows = report.get('public_scaler_comparison', [])
    encoder_rows = report.get('input_encoder_comparison', [])
    if encoder_rows:
        with (output_dir / 'input_encoder_comparison.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(encoder_rows[0]))
            writer.writeheader()
            writer.writerows(encoder_rows)
    if public_rows:
        with (output_dir / 'public_scaler_comparison.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(public_rows[0]))
            writer.writeheader()
            writer.writerows(public_rows)
    if comparison:
        with (output_dir / 'training_reference_comparison.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    lines = ['# DAGPipe scale diagnosis', '', f"Task: {report['task']}",
             f"Format: {report['format']}, space: {report['operator_space']}, scale: {report['scale_protocol']}",
             f"Query: {report['query_source']}, rows: {report['query_rows']}", '',
             '## First finite threshold crossings', '']
    for key, hit in report['first_threshold_crossings'].items():
        lines.append(f"- {key}: event {hit['event']} {hit['stage']}, layer {hit['layer']}, "
                     f"node {hit['node']} ({hit['op']}), columns {', '.join(hit['columns'][:12])}")
    if not report['first_threshold_crossings']:
        lines.append('- None')
    if 'matches_logged_abs_max' in report:
        lines += ['', f"Logged abs_max: {report['expected_abs_max']}, matches: {report['matches_logged_abs_max']}"]
    final = report['events'][-1]['query']['summary']
    lines += ['', f"Full query output abs_max: {report['full_query_output_abs_max']}",
              f"Reference statistics changes across batches: {report['reference_statistics_changes_across_batches']}",
              f"First-batch final NaN: {final['nan_count']}, Inf: {final['inf_count']}", '']
    for peak in report['robust_scaler_peaks']:
        lines.append(f"- Robust layer {peak['layer']} node {peak['node']} {peak['split']}: "
                     f"{peak['column']} row {peak['row_position']}, "
                     f"{peak['scaler_input']} -> {peak['scaler_output']}, "
                     f"IQR {peak['iqr']}, denominator {peak['denominator']}")
    lines += [
              '', 'Event distributions cover the first batch. Peak locations and threshold scans cover all queries.',
              'Node and layer indices are zero-based. Quantiles summarize finite values only.',
              'Match scaler and node rows by layer/node/column. Edge rows report the actual weighted contribution.']
    if comparison:
        lines += ['', '## Training reference coverage (read-only)',
                  f"Training rows: {report['comparison_training_rows']}",
                  'Statistics below use the original checkpoint at every upstream node; no refitting was performed.',
                  'These are coverage diagnostics, not predictions of a retrained model or its AUC.']
        ranked = sorted(comparison, key=lambda r: r['train_to_reference_iqr_ratio'] or 0, reverse=True)
        for row in ranked[:10]:
            lines.append(f"- Layer {row['layer']} node {row['node']} {row['op']} {row['column']}: "
                         f"reference IQR={row['reference_iqr']}, training IQR={row['training_iqr']}, "
                         f"ratio={row['train_to_reference_iqr_ratio']}, outside reference range={row['outside_reference_range_fraction']}")
    if public_rows:
        lines += ['', '## Same-input scaler comparison',
                  'All operators receive the exact routed node input and original reference prefix.',
                  'Public outputs retain float32; tolerance is rtol=2e-6, atol=1e-6.',
                  'Old results use isolated dagpipe_old operators, NOT a replay of the old DAG.',
                  f"Current/public mismatched values: {sum(r['current_public_mismatches'] for r in public_rows)}",
                  f"Current/old mismatched values: {sum(r['current_old_mismatches'] for r in public_rows)}",
                  f"Nonfinite values across comparisons: {sum(r['nonfinite_values'] for r in public_rows)}"]
    (output_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def compare_training_reference(payload, training_frame, *, device='cpu', v2_protocol=None):
    """Observe real scaler inputs, without replacing any frozen reference values.

    CPU storage is proportional to training rows * features * scaler nodes.
    No query labels are inspected. Caller must supply only legal training rows.
    """
    # Share the strict historical-format validation with the ordinary diagnostic.
    trace_checkpoint(payload, v2_protocol=v2_protocol, device=device)
    replay = dict(payload)
    if payload['format'].endswith('.v2'):
        replay['scale_protocol'] = V2_PROTOCOLS[v2_protocol]
    frozen = FrozenTransform(replay, device=device)
    encoded = frozen.encoder.transform(training_frame)
    if len(encoded) < 2:
        raise ValueError('Training comparison requires at least two legal training rows')
    n_ref = len(frozen.reference)
    blocks, refs, denominators = {}, {}, {}
    scaler_names = {'StandardScaler', 'RobustScaler', 'MinMaxScaler'}

    def observe(stage, value, layer=None, node=None, **details):
        if node is None or payload['layers'][layer][node] not in scaler_names:
            return
        key = (layer, node)
        if stage == 'after_routing':
            array = value.detach().cpu().numpy()
            refs.setdefault(key, array[:n_ref].copy())
            blocks.setdefault(key, []).append(array[n_ref:].copy())
        elif stage == 'scaler_statistics':
            denominators.setdefault(key, details['denominator'].detach().cpu().numpy().ravel().copy())

    frozen.model.forward_observer = observe
    try:
        with torch.no_grad():
            for start in range(0, len(encoded), frozen.batch_size):
                query = frozen.reference.new_tensor(encoded[start:start + frozen.batch_size])
                reference_forward(frozen.model, query, frozen.reference)
    finally:
        del frozen.model.forward_observer
    rows = []
    for (layer, node), chunks in blocks.items():
        values = np.concatenate(chunks)
        for i, spec in enumerate(payload['encoder']):
            ref, train = refs[layer, node][:, i], values[:, i]
            ref, train = ref[np.isfinite(ref)], train[np.isfinite(train)]
            if not len(ref) or not len(train):
                raise ValueError('No finite scaler inputs for training comparison')
            rq, tq = np.percentile(ref, [25, 50, 75]), np.percentile(train, [25, 50, 75])
            riqr, tiqr = float(rq[2] - rq[0]), float(tq[2] - tq[0])
            rows.append(dict(layer=layer, node=node, op=payload['layers'][layer][node], column=spec['column'],
                             reference_rows=n_ref, training_rows=len(encoded),
                             reference_q25=float(rq[0]), reference_median=float(rq[1]), reference_q75=float(rq[2]),
                             reference_iqr=riqr, actual_denominator=float(denominators[layer, node][i]),
                             training_q25=float(tq[0]), training_median=float(tq[1]), training_q75=float(tq[2]),
                             training_iqr=tiqr, train_to_reference_iqr_ratio=tiqr / riqr if riqr > 0 else None,
                             training_abs_max=float(np.abs(train).max()),
                             outside_reference_range_fraction=float(((train < ref.min()) | (train > ref.max())).mean())))
    return rows


def restore_training_availability(frame, classes_csv):
    """Restore only training eligibility from the original two-column CSV.

    Scan IDs first; parse a class only for a frozen, chronological training
    candidate. Labels are discarded immediately after testing availability.
    """
    from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
    data = EllipticBitcoinData()
    probe = frame.copy()
    data._normalize_tx_id_column(probe, 'txId')
    probe['txId'] = probe['txId'].astype(str)
    # Reuse the dataset mask to select chronological candidates without labels.
    probe['class'] = 0
    data.train_df = probe
    data._prepare_validation()
    candidates = set(data.train_df.loc[data.train_df['__split__'].eq('train'), 'txId'])
    available = {}
    with Path(classes_csv).open(encoding='utf-8-sig', newline='') as stream:
        if next(csv.reader([stream.readline()]), None) != ['txId', 'class']:
            raise ValueError('Expected original Elliptic CSV header txId,class')
        for line in stream:
            identifier, separator, rest = line.partition(',')
            identifier = identifier.strip()
            if identifier not in candidates:
                continue
            if not separator or identifier in available:
                raise ValueError('Malformed or duplicate training ID in Elliptic classes CSV')
            label = EllipticBitcoinData._normalize_class_value(next(csv.reader([rest]))[0])
            if pd.isna(label):
                raise ValueError(f'Unrecognized training class for txId {identifier}')
            available[identifier] = 0 if label >= 0 else -1
    missing = candidates - available.keys()
    if missing:
        raise ValueError(f'Classes CSV is missing {len(missing)} chronological training IDs')
    probe['class'] = probe.txId.map(available).fillna(-1).astype(int)
    return probe


def load_elliptic_training_features(directory, columns, classes_csv=None):
    """Use the dataset's actual chronological masks; never open std-test labels."""
    import pyarrow.parquet as pq
    path = directory / 'train_frozen.parquet'
    available = pq.ParquetFile(path).schema_arrow.names
    if 'class' not in available and classes_csv is None:
        raise ValueError('train_frozen.parquet lacks class needed to identify supervised training rows. '
                         'Supply --elliptic-classes-csv pointing to elliptic_txs_classes.csv, '
                         'or use --training-features-parquet exported from graph_splits(data)["train"]; '
                         'the entire frozen file is NOT a training-only split.')
    names = list(dict.fromkeys(['txId', 'time_step', *(['class'] if 'class' in available else []), *columns,
                               *(['__split__'] if '__split__' in available else [])]))
    frame = pd.read_parquet(path, columns=names)
    if 'class' not in frame:
        frame = restore_training_availability(frame, classes_csv)
    return select_elliptic_training_features(frame, columns)


def select_elliptic_training_features(frame, columns):
    from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
    if 'class' not in frame:
        raise ValueError('Training split selection requires class availability')
    data = EllipticBitcoinData()
    data.train_df = data._normalize_frozen_frame(frame)
    data._prepare_validation()
    return data.train_df.loc[data.train_df['__split__'].eq('train'), columns].reset_index(drop=True)


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cpu', help='Use the original device type for closest numerical replay')
    parser.add_argument('--elliptic-frozen-dir', type=Path, help='Directory containing train_frozen.parquet and std_test.parquet')
    parser.add_argument('--graph-phase', choices=['hpo', 'final'], default='final')
    parser.add_argument('--expected-abs-max', type=float)
    parser.add_argument('--query-parquet', type=Path, help='Prepared validation feature Parquet; only checkpoint columns are read')
    parser.add_argument('--max-query-rows', type=int, default=1024)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--v2-scale-protocol', choices=V2_PROTOCOLS)
    parser.add_argument('--compare-training-reference', action='store_true',
                        help='Compare frozen references with legal training rows; does not refit')
    parser.add_argument('--compare-public-scalers', action='store_true',
                        help='Compare public ScaleFeature and dagpipe_old scalers on exact replay node inputs')
    parser.add_argument('--training-features-parquet', type=Path,
                        help='Explicit training-only export from graph_splits(data)["train"], never the full frozen file')
    parser.add_argument('--elliptic-classes-csv', type=Path,
                        help='Original txId,class CSV for legacy frozen files; only training candidate classes are parsed')
    args = parser.parse_args()
    if args.training_features_parquet and not args.compare_training_reference:
        parser.error('--training-features-parquet requires --compare-training-reference')
    if args.elliptic_classes_csv and (not args.compare_training_reference or not args.elliptic_frozen_dir
                                    or args.training_features_parquet):
        parser.error('--elliptic-classes-csv requires training comparison with --elliptic-frozen-dir only')
    if args.max_query_rows < 1:
        parser.error('--max-query-rows must be positive')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    raw = args.checkpoint.read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict) or payload.get('format') not in {
        'dppbench.DAGPipe.reference.v2', 'dppbench.DAGPipe.reference.v3'
    }:
        parser.error('Unsupported checkpoint format. Use the original implementation for historical formats other than reference.v2/v3.')
    frame = None
    if args.elliptic_frozen_dir and args.query_parquet:
        parser.error('Choose either --elliptic-frozen-dir or --query-parquet')
    if args.elliptic_frozen_dir:
        if payload.get('task') != 'elliptic_bitcoin':
            parser.error('--elliptic-frozen-dir requires an Elliptic checkpoint')
        columns = [spec['column'] for spec in payload['encoder']]
        if {'class', 'label', 'target'} & set(columns):
            parser.error('Checkpoint feature columns contain a label')
        files = ['train_frozen.parquet']
        if args.graph_phase == 'final':
            files.append('std_test.parquet')
        frames = [pd.read_parquet(args.elliptic_frozen_dir / name, columns=['txId', *columns]) for name in files]
        expected = [194457, 9312][:len(files)]
        if [len(part) for part in frames] != expected:
            parser.error(f'Frozen feature row counts differ from this experiment: expected {expected}. '
                         'Provide exact replay features via --query-parquet if label filtering changed membership.')
        frame = pd.concat(frames, ignore_index=True)
        if frame.txId.astype(str).duplicated().any():
            parser.error('Duplicate or overlapping frozen transaction IDs')
    if args.query_parquet:
        import pyarrow.parquet as pq
        columns = [spec['column'] for spec in payload['encoder']]
        parquet = pq.ParquetFile(args.query_parquet)
        if 'txId' in parquet.schema_arrow.names and 'txId' not in columns:
            columns.append('txId')
        batch = next(parquet.iter_batches(batch_size=args.max_query_rows, columns=columns), None)
        if batch is None:
            parser.error('Validation feature file has no rows')
        frame = batch.to_pandas()
    try:
        report, _ = trace_checkpoint(payload, frame, v2_protocol=args.v2_scale_protocol, device=args.device,
                                     compare_public_scalers=args.compare_public_scalers)
    except (ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    report.update(checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                  validation_file=str(args.query_parquet) if args.query_parquet else None)
    if args.compare_public_scalers:
        from scripts.dagpipe_scaler_comparison import old_runtime
        import baselines.DAGPipe.runtime as current_runtime
        import dppbench.operators.preprocessing.scale_feature as public_module
        report['comparison_source_sha256'] = {
            module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (old_runtime, current_runtime, public_module)}
    if args.elliptic_frozen_dir:
        report.update(query_source=f'elliptic_frozen_{args.graph_phase}', frozen_dir=str(args.elliptic_frozen_dir),
                      input_files=files, input_file_rows=[len(part) for part in frames])
    if args.expected_abs_max is not None:
        report['expected_abs_max'] = args.expected_abs_max
        report['matches_logged_abs_max'] = bool(np.isclose(report['full_query_output_abs_max'],
                                                          args.expected_abs_max, rtol=5e-6))
    report['device'] = args.device
    if args.compare_training_reference:
        columns = [spec['column'] for spec in payload['encoder']]
        try:
            if args.training_features_parquet:
                if args.training_features_parquet.name in {'train_frozen.parquet', 'std_test.parquet'}:
                    raise ValueError('Use a training-only export, not a whole frozen graph file')
                training = pd.read_parquet(args.training_features_parquet, columns=columns)
                source = str(args.training_features_parquet)
            elif args.elliptic_frozen_dir:
                training = load_elliptic_training_features(args.elliptic_frozen_dir, columns, args.elliptic_classes_csv)
                source = 'EllipticBitcoinData._prepare_validation train mask'
                if args.elliptic_classes_csv:
                    report['training_classes_csv'] = str(args.elliptic_classes_csv)
            else:
                raise ValueError('Training comparison requires --elliptic-frozen-dir or --training-features-parquet')
            report['training_reference_comparison'] = compare_training_reference(
                payload, training, device=args.device, v2_protocol=args.v2_scale_protocol)
            if args.compare_public_scalers:
                from scripts.dagpipe_scaler_comparison import compare_input_encoders
                report['input_encoder_comparison'] = compare_input_encoders(payload, training, frame)
            report.update(comparison_training_rows=len(training), comparison_training_source=source)
        except (ValueError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
    write_reports(report, args.output_dir)
    print(f'Reports saved to {args.output_dir}')


if __name__ == '__main__':
    main()
