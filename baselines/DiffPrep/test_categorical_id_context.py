"""Categorical IDs retained by the real model need fixed train-only encoding."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from baselines.common.pipeline import PipelineStep, Pipeline, check_pipeline
from baselines.common.op_spec import OpCategory
from baselines.DiffPrep.slot_planner import Slot
from baselines.DiffPrep.search_space import ContinuousPipeline
from baselines.DiffPrep.tabular_bank import TabularTransformBank
from dppbench.dataset import TabularData
from dppbench.models.lightgbm_model import LightGBMModel


@pytest.mark.parametrize('dtype', ['object', 'string', 'category'])
@pytest.mark.parametrize('flex', [False, True])
def test_station_id_fixed_context_matches_export_and_trains_lightgbm(dtype, flex):
    frame = pd.DataFrame({'start_station_id': pd.Series(['s1', 's2'] * 16 + ['future'] * 8, dtype=dtype),
                          '_hour_ts': pd.date_range('2024-01-01', periods=40, freq='h'),
                          '_hour_idx': np.arange(40), 'hour_of_day': np.arange(40) % 24,
                          'rental_count': np.arange(40) % 7})
    ctx = SimpleNamespace(id_col='start_station_id', time_col='_hour_idx', target_col='rental_count')
    data = TabularData('citibike_jc_hourly')
    slots = [Slot(0, OpCategory.CLEANING, ['HandleMV'], 'soft', True, 'HandleMV'),
             Slot(1, OpCategory.PREPROCESSING, ['LabelEncode'], 'soft', True, 'LabelEncode')]
    def make_step(name, *_):
        return PipelineStep(name, params={'method': 'median', 'action': 'impute'} if name == 'HandleMV' else {})
    with patch('baselines.DiffPrep.tabular_bank.diffprep_make_step', side_effect=make_step):
        bank = TabularTransformBank(data, ctx, slots, frame, np.arange(32), ['hour_of_day'], 42)
    cont = ContinuousPipeline(slots, n_features=1, feature_names=['hour_of_day'], flex=flex)
    cont.tabular_bank = bank
    output = cont(*bank.inputs('cpu'))
    projected = bank.projected_frame(cont, output)
    steps = bank.selected_steps(cont)
    assert bank.names == ['hour_of_day']  # ID is not a new trainable feature gate.
    assert bank.context_categories['start_station_id'] == {'s1': 0, 's2': 1}
    assert projected.start_station_id.iloc[32:].eq(-1).all()
    assert next(s for s in steps if s.op == 'LabelEncode').params['cols'] == ['start_station_id']
    assert check_pipeline(Pipeline(steps=steps).to_yaml()).valid
    fit, query = frame.iloc[:32].copy(), frame.copy()
    for step in steps:
        op = data._load_op(step.op)(**step.params)
        fit = op.transform(fit)
        query = op.transform(query)
    np.testing.assert_array_equal(projected.start_station_id, query.start_station_id)
    assert pd.api.types.is_numeric_dtype(query.start_station_id)
    assert bank.contract()['fixed_categorical_context'] == ['start_station_id']
    X = query.drop(columns='rental_count')
    model = LightGBMModel(task='regression', n_estimators=5, min_child_samples=1)
    model.fit(X.iloc[:32], frame.rental_count.iloc[:32], X.iloc[32:], frame.rental_count.iloc[32:])
    assert np.isfinite(model.predict(X.iloc[32:])).all()
