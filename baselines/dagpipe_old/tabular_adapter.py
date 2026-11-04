"""Fixed, label-free side-table inputs shared by search and checkpoint replay."""
from __future__ import annotations

import pandas as pd


def tabular_adapter(task):
    # Account metadata alone omits Berka's historical financial information.
    table, key, method = {
        'berka': ('account', 'account_id', 'key'),
        'default_credit': ('monthly_history', 'ID', 'agg'),
    }.get(task, (None, None, None))
    adapter = {'task_type': 'tabular', 'table': table, 'key': key,
               'method': method, 'ops': ['JoinTable'] if table else []}
    if task == 'berka':
        adapter['berka_history_version'] = 1
        adapter['ops'].append('JoinTable')
    return adapter


def _berka_dates(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        result = pd.to_datetime(series, errors='raise')
    else:
        values = series.astype('string').str.replace(r'\.0$', '', regex=True)
        if not values.str.fullmatch(r'\d{6}').fillna(False).all():
            raise ValueError('Berka history requires valid YYMMDD dates')
        result = pd.to_datetime(values, format='%y%m%d', errors='raise')
    if result.isna().any():
        raise ValueError('Berka history dates cannot be missing')
    return result


def _join_berka_history(frame, transactions):
    """Aggregate strictly pre-loan events independently for each main-table row.

    Recheck the cutoff even if the loader already filtered trans. No target or
    other loan row is consulted. Same-day events are excluded (no event times).
    """
    required = {'account_id', 'date'}
    if not required.issubset(frame) or not required.issubset(transactions):
        raise ValueError('Berka history requires account_id/date in loans and trans')
    if frame['account_id'].isna().any():
        raise ValueError('Berka loans require nonmissing account_id')
    numeric = [c for c in ('amount', 'balance') if c in transactions]
    categorical = [c for c in ('type', 'operation', 'k_symbol', 'bank') if c in transactions]
    if not numeric:
        raise ValueError('Berka trans requires amount or balance financial features')
    loans = frame[['account_id']].reset_index(drop=True).copy()
    loans['__loan_row'] = range(len(loans))
    loans['__loan_date'] = _berka_dates(frame['date']).to_numpy()
    events = transactions[['account_id', *numeric, *categorical]].copy()
    events['__event_date'] = _berka_dates(transactions['date']).to_numpy()
    for col in numeric:
        events[col] = pd.to_numeric(events[col], errors='coerce')
    history = loans.merge(events, on='account_id', how='inner', validate='many_to_many')
    history = history.loc[history['__event_date'] < history['__loan_date']]
    grouped = history.groupby('__loan_row', sort=False)
    features = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    features['trans__count'] = grouped.size().reindex(features.index, fill_value=0)
    for col in numeric:
        for stat in ('mean', 'max', 'min'):
            features[f'trans__{col}__{stat}'] = grouped[col].agg(stat)
    for col in categorical:
        features[f'trans__{col}__nunique'] = grouped[col].nunique()
    if set(features) & set(frame):
        raise ValueError('Berka history feature names collide with main-table columns')
    return pd.concat([frame.reset_index(drop=True), features], axis=1)


def augment_splits(data, splits, adapter):
    table = adapter.get('table')
    if not table:
        return splits
    aux = data.auxiliary_dfs.get(table)
    key = adapter['key']
    if aux is None or aux.empty or key not in aux:
        raise ValueError(f'DAGPipe requires nonempty auxiliary table {table} with key {key}')
    aux = aux.drop(columns=[data.target_col, '__split__', 'sample_weight'], errors='ignore').copy()
    if adapter['method'] == 'agg':
        numeric = [c for c in aux if c != key and pd.api.types.is_numeric_dtype(aux[c])]
        if not numeric:
            raise ValueError(f'DAGPipe auxiliary table {table} has no numeric history')
        right = aux.groupby(key)[numeric].agg(['mean', 'max', 'min'])
        right.columns = [f'{table}__{c}__{stat}' for c, stat in right.columns]
        right = right.reset_index()
    else:
        if aux[key].duplicated().any():
            raise ValueError(f'DAGPipe {table}.{key} must be unique')
        right = aux.rename(columns={c: f'{table}__{c}' for c in aux if c != key})
    result = {}
    for name, frame in splits.items():
        if frame is None:
            result[name] = None
            continue
        overlap = (set(frame) & set(right)) - {key}
        if overlap:
            raise ValueError(f'DAGPipe auxiliary feature collision: {sorted(overlap)}')
        result[name] = frame.merge(right, on=key, how='left', sort=False, validate='many_to_one')
    if adapter.get('berka_history_version'):
        if adapter['berka_history_version'] != 1:
            raise ValueError('Unsupported Berka history protocol')
        transactions = data.auxiliary_dfs.get('trans')
        if transactions is None:
            raise ValueError('DAGPipe Berka requires trans; account-only fallback is disabled')
        result = {name: None if frame is None else _join_berka_history(frame, transactions)
                  for name, frame in result.items()}
    return result
