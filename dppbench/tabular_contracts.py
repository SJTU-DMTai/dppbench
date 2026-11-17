"""Tabular-only data contracts. No recommendation protocol or guessed clocks."""
import hashlib
import numpy as np
import pandas as pd


def dataset_key(data):
    module = type(data).__module__.split('.')
    if 'tasks' in module:
        return module[module.index('tasks') + 1]
    return getattr(data, 'name', '')


def diagnostic_columns(columns, target=None):
    exact = {'sample_weight', '__split__', '__source_row__', '__row_idx__'}
    if target:
        exact.add(f'{target}_corrected')
    return [c for c in columns if c in exact or str(c).startswith('__dppbench_')]


def model_features(frame, target, id_col=None):
    return frame.drop(columns=[target, id_col, *diagnostic_columns(frame.columns, target)], errors='ignore')


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def datetime_seconds(series):
    """Explicit Unix seconds independent of pandas' s/ms/us/ns storage dtype."""
    values = pd.to_datetime(series, errors='coerce')
    array = values.array
    if hasattr(array, 'as_unit'):
        array = array.as_unit('ns')
    numeric = np.asarray(array.asi8, dtype=np.float64) / 1e9
    return pd.Series(np.where(values.notna(), numeric, 0.), index=series.index)


def time_rule(dataset, column):
    aliases = {'BeijingAirQuality': 'beijing_air_quality', 'Berka': 'berka',
               'BikeSharing': 'bike_sharing', 'CitibikeJcHourly': 'citibike_jc_hourly',
               'NycTaxiHourly': 'nyc_taxi_hourly'}
    dataset = aliases.get(dataset, dataset)
    rules = {
        'beijing_air_quality': {'_sort_idx': {'format': '%Y%m%d%H'}},
        'berka': {'date': {'format': '%y%m%d'}, 'issued': {'format': '%y%m%d'}},
        'bike_sharing': {'dteday': {'format': '%Y-%m-%d'},
                         '_sort_idx': {'source_date': 'dteday', 'source_hour': 'hr', 'format': '%Y-%m-%d'}},
        'citibike_jc_hourly': {'_hour_idx': {'unit': 's', 'origin': 'unix'}, '_hour_ts': {'format': 'mixed'}},
        'nyc_taxi_hourly': {'_hour_idx': {'unit': 's', 'origin': 'unix'}, '_hour_ts': {'format': 'mixed'}},
    }
    return rules.get(dataset, {}).get(column)


def parse_time(series, rule=None, frame=None):
    """Numeric dates need an explicit format or unit AND origin."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors='coerce')
    rule = rule or {}
    if 'source_date' in rule:
        if frame is None or any(c not in frame for c in (rule['source_date'], rule['source_hour'])):
            raise ValueError('Calendar composition requires declared date/hour source columns')
        dates = pd.to_datetime(frame[rule['source_date']], format=rule['format'], errors='coerce')
        hours = pd.to_numeric(frame[rule['source_hour']], errors='coerce')
        hours = hours.where(hours.between(0, 23) & hours.eq(np.floor(hours)))
        return dates + pd.to_timedelta(hours, unit='h')
    if 'unit' in rule:
        if 'origin' not in rule:
            raise ValueError('Numeric time requires an explicit origin')
        return pd.to_datetime(pd.to_numeric(series, errors='coerce'), unit=rule['unit'], origin=rule['origin'], errors='coerce')
    if 'format' in rule:
        values = series
        if pd.api.types.is_numeric_dtype(series):
            values = series.where(np.isfinite(series) & series.eq(np.floor(series))).astype('Int64').astype('string')
            if rule['format'] == '%y%m%d':
                values = values.str.zfill(6)
        return pd.to_datetime(values, format=rule['format'], errors='coerce')
    if pd.api.types.is_numeric_dtype(series):
        raise ValueError('Numeric time has no declared format/unit/origin; protocol confirmation required')
    return pd.to_datetime(series, format='mixed', errors='coerce')


def fixed_search_view(data, splits, small_n, seed):
    """Preserve original TOTAL search pool cap; never sample before splitting.

    Search validation is a fixed subset of canonical HPO validation. The HPO
    partition itself is unchanged. Its hash is retained before sampling.
    """
    out = dict(splits)
    data._tabular_split_contract = {
        'version': 'tabular_fixed_boundary_v1',
        'canonical_train_sha256': frame_hash(splits['train']),
        'canonical_val_sha256': frame_hash(splits['val']),
        'canonical_train_rows': len(splits['train']), 'canonical_val_rows': len(splits['val']),
        'small_n_total_cap': int(small_n or 0), 'seed': int(seed),
    }
    total = len(splits['train']) + len(splits['val'])
    if small_n and total > small_n:
        if small_n < 2:
            raise ValueError('small_n must fit nonempty train and validation')
        nval = max(1, min(len(splits['val']), int(np.ceil(small_n * len(splits['val']) / total))))
        ntrain = min(len(splits['train']), small_n - nval)
        if ntrain < 1:
            raise ValueError('small_n leaves no training rows')
        for name, n in [('train', ntrain), ('val', nval)]:
            out[name] = splits[name].sample(n=n, random_state=seed).sort_index().reset_index(drop=True)
    data._tabular_split_contract.update(search_train_rows=len(out['train']), search_val_rows=len(out['val']))
    return out


def prepare_references(data, train):
    hook = getattr(data, 'fit_tabular_references', None)
    if hook:
        hook(train)
