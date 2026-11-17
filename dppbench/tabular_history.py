"""Strict tabular query history; never uses held-out rows as history by default."""
import numpy as np
import pandas as pd


def positive_counts(values, name):
    values = values if isinstance(values, list) else [values]
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in values):
        raise ValueError(f'{name} must contain strictly positive integers')
    return values


def history_features(query, history, op):
    if op.group_cols is None or not op.time_col:
        raise ValueError('History requires explicit entity keys (or [] for a declared single series) and time_col; protocol unresolved')
    keys = list(op.group_cols)
    required = list(dict.fromkeys(keys + [op.time_col, op.target_col]))
    available = getattr(op, 'available_at_col', None)
    if not available and getattr(op, 'availability', None) != 'event_time':
        raise ValueError('History availability is unresolved: declare available_at_col or confirm availability=event_time')
    if available:
        required.append(available)
    for frame in (query, history):
        if any(c not in frame for c in required):
            raise ValueError('History entity/time/value columns missing; no ungrouped fallback')
        if frame[keys + [op.time_col]].isna().any().any():
            raise ValueError('History entity/time contains null values')
    result = query.copy()
    # Stable sort preserves observation-lag semantics for STRICTLY earlier rows.
    # All equal-time rows are excluded together, independent of input ordering.
    groups = history.groupby(keys, sort=False, dropna=False) if keys else [((), history)]
    lookup = {}
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        part = part.sort_values(op.time_col, kind='mergesort')
        lookup[key] = (pd.Index(part[op.time_col]),pd.to_numeric(part[op.target_col], errors='raise'),
            part[available].reset_index(drop=True) if available else None)
    lag = hasattr(op, 'lags')
    sizes = op.lags if lag else op.windows
    output = {f'{op.target_col}_lag_{n}': [] for n in sizes} if lag else {
        f'{op.target_col}_roll_{agg}_{n}': [] for n in sizes for agg in op.aggs}
    for _, row in query.iterrows():
        key = tuple(row[c] for c in keys)
        times, values, available_values = lookup.get(key, (np.array([]), pd.Series(dtype=float), None))
        end = int(times.searchsorted(row[op.time_col], side='left')) if len(times) else 0
        needed = max(sizes)
        if available_values is None:
            values = values.iloc[max(0, end-needed):end]
        else:
            positions = []
            cursor = end - 1
            while cursor >= 0 and len(positions) < needed:
                at = available_values.iloc[cursor]
                if pd.notna(at) and at <= row[op.time_col]:
                    positions.append(cursor)
                cursor -= 1
            values = values.iloc[list(reversed(positions))]
        end = len(values)
        for n in sizes:
            if lag:
                output[f'{op.target_col}_lag_{n}'].append(values.iloc[end-n] if end >= n else np.nan)
            else:
                prior = values.iloc[max(0, end-n):end]
                for agg in op.aggs:
                    value = getattr(prior, agg)() if prior.notna().any() else np.nan
                    output[f'{op.target_col}_roll_{agg}_{n}'].append(value)
    for name, values in output.items():
        result[name] = values
    return result


def transform_partitioned_history(data, frame, op):
    from dppbench.tabular_contracts import time_rule, parse_time, dataset_key
    from .time_applicability import resolve_time_params
    cfg = getattr(data, 'model_cfg', {}) or {}
    contract = cfg.get('history_protocol', cfg.get('tabular_history', {}))
    params, errors = resolve_time_params(op.name, {
        k: getattr(op, k, None) for k in ('target_col', 'group_cols', 'time_col', 'availability', 'available_at_col')
    }, dataset_key(data), cfg.get('feature', {}).get('time_rules', {}), contract, data.target_col)
    if errors:
        raise ValueError('; '.join(errors))
    for k, v in params.items():
        setattr(op, k, v)
    contract = dict(contract, **contract.get('sources', {}).get(op.target_col, {}))
    from dppbench.feature_contracts import target_feature_errors
    errors = target_feature_errors(op.name, vars(op), data.target_col)
    if errors:
        raise ValueError('; '.join(errors))
    work = frame.copy()
    if '__split__' not in work:
        raise ValueError('History training source requires explicit partition membership')
    if not op.time_col:
        raise ValueError('History time_col must be declared')
    original_time = work[op.time_col].copy()
    from .time_applicability import declared_clock
    rule = contract.get('time_rule') or declared_clock(dataset_key(data), op.time_col, cfg.get('feature', {}).get('time_rules', {}))
    if rule is not None and rule.get('kind') != 'ordered_step':
        work[op.time_col] = parse_time(work[op.time_col], rule, work)
    elif not (pd.api.types.is_numeric_dtype(work[op.time_col]) or pd.api.types.is_datetime64_any_dtype(work[op.time_col])):
        raise ValueError('History time format is unresolved; declare tabular_history.time_rule')
    if op.available_at_col:
        original_available = work[op.available_at_col].copy()
        arule = declared_clock(dataset_key(data), op.available_at_col, cfg.get('feature', {}).get('time_rules', {}))
        if arule.get('kind') != 'ordered_step':
            work[op.available_at_col] = parse_time(work[op.available_at_col], arule, work)
    history = work.loc[work['__split__'].eq('train')] if '__split__' in work else work
    result = history_features(work, history, op)
    result[op.time_col] = original_time
    if op.available_at_col:
        result[op.available_at_col] = original_available
    return result
