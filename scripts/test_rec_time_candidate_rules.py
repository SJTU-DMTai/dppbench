"""Reject undeclared numeric clocks before candidate training."""
import random

import pandas as pd
import pytest

from baselines.common.data_ctx import DataContext, TableContext, _infer_rec_context
from baselines.common.pipeline import build_default_params, validate_step_params
from dppbench.operators.integration.parse_date import ParseDate
from dppbench.operators.feature_engineering.extract_date_time_feature import ExtractDateTimeFeature


def context(name='unknown_clock', dtype='float64', rules=None):
    return DataContext('rec', name, 'interaction', tables={
        'interaction': TableContext('interaction', 'primary',
            dtypes={'timestamp': dtype}, configured_time_col='timestamp')},
        time_rules=rules or {})


@pytest.mark.parametrize('op', ['ParseDate', 'ExtractDateTimeFeature'])
def test_unknown_numeric_clock_is_skipped_and_manual_candidate_rejected(op):
    ctx = context()
    assert build_default_params(op, ctx, random.Random(42)) is None
    assert any('declared' in e for e in validate_step_params(op, {'cols': ['timestamp']}, ctx))
    assert any('declared' in e for e in validate_step_params(op, {
        'cols': ['timestamp'], 'time_rules': {'timestamp': {'unit': 's'}}}, ctx))


@pytest.mark.parametrize('op', ['ParseDate', 'ExtractDateTimeFeature'])
@pytest.mark.parametrize('name,rules,unit', [
    ('kuairec', {'timestamp': {'unit': 'ms', 'origin': 'unix'}}, 'ms'),
    ('movielens', {}, 's'), ('AmazonBeauty', {}, 's'), ('Yelp', {}, 's'),
    ('kuairec', {}, 's'), ('KuaiRec', {}, 's')])
def test_declared_clock_candidate_executes(op, name, rules, unit):
    ctx = context(name, rules=rules)
    params = build_default_params(op, ctx, random.Random(42))
    assert params['time_rules']['timestamp'] == {'unit': unit, 'origin': 'unix'}
    assert validate_step_params(op, params, ctx) == []
    frame = pd.DataFrame({'timestamp': [1500000000 * (1000 if unit == 'ms' else 1)]})
    result = {'ParseDate': ParseDate, 'ExtractDateTimeFeature': ExtractDateTimeFeature}[op](**params).transform(frame)
    if op == 'ParseDate':
        assert result.timestamp.dt.year.iloc[0] == 2017
    else:
        assert result.timestamp_year.iloc[0] == 2017


@pytest.mark.parametrize('dtype', ['object', 'datetime64[ns]'])
@pytest.mark.parametrize('op', ['ParseDate', 'ExtractDateTimeFeature'])
def test_nonnumeric_clock_remains_available(dtype, op):
    ctx = context(dtype=dtype, rules={'timestamp': {'kind': 'calendar', 'format': 'mixed'}})
    assert build_default_params(op, ctx, random.Random(42)) is not None
    assert validate_step_params(op, {'cols': ['timestamp']}, ctx) == []


def test_context_carries_configured_time_rules():
    from types import SimpleNamespace
    rules = {'timestamp': {'unit': 'ms', 'origin': 'unix'}}
    data = SimpleNamespace(interaction_df=pd.DataFrame({'timestamp': [1500000000000]}),
        _user_id_col='user_id', _item_id_col='item_id', user_df=None, item_df=None,
        model_cfg={'feature': {'time_rules': rules}}, col_types={'timestamp': 'timestamp'})
    ctx = _infer_rec_context('kuairec', data)
    assert ctx.time_rules == rules
    ctx.time_rules['timestamp']['unit'] = 's'
    assert rules['timestamp']['unit'] == 'ms'


@pytest.mark.parametrize('op', ['ParseDate', 'ExtractDateTimeFeature'])
def test_kuairec_fractional_unix_seconds_and_missing_values(op):
    ctx = context('kuairec')
    params = build_default_params(op, ctx, random.Random(42))
    frame = pd.DataFrame({'timestamp': [1593878903.438, float('nan')]})
    result = {'ParseDate': ParseDate, 'ExtractDateTimeFeature': ExtractDateTimeFeature}[op](**params).transform(frame)
    if op == 'ParseDate':
        assert abs(result.timestamp.iloc[0] - pd.Timestamp('2020-07-04 16:08:23.438')) < pd.Timedelta(microseconds=1)
        assert pd.isna(result.timestamp.iloc[1])
    else:
        assert result.timestamp_year.iloc[0] == 2020
        assert pd.isna(result.timestamp_year.iloc[1])
