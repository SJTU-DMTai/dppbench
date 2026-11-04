"""Small, data-free regression suite: python -m unittest scripts.test_time_protocol_applicability."""
import random
import unittest
from types import SimpleNamespace

import pandas as pd
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import (Pipeline, PipelineStep, make_step,
    build_default_params, validate_step_params, check_pipeline, assign_dag_structure)
from dppbench.time_applicability import resolve_time_params
from dppbench.operators.integration.parse_date import ParseDate
from dppbench.operators.feature_engineering.create_lag_feature import CreateLagFeature
from dppbench.tabular_history import transform_partitioned_history


def context(task='graph', name='elliptic_bitcoin', rules=None, history=None):
    return DataContext(task, name, 'main', tables={'main': TableContext('main', 'primary',
        dtypes={'time_step': 'int64', 'value': 'float64', 'entity': 'object', 'label': 'int64'},
        configured_time_col='time_step', configured_target_col='label')},
        time_rules=rules or {}, history_protocol=history or {})


class TimeProtocolTests(unittest.TestCase):
    def test_elliptic_generation_and_mutated_params_rejected(self):
        ctx = context()
        for op in ('ParseDate', 'ExtractDateTimeFeature', 'CreateLagFeature', 'CreateRollingFeature'):
            self.assertIsNone(make_step(op, ctx, random.Random(42)))
            self.assertIsNone(build_default_params(op, ctx, random.Random(42)))
            params = {'cols': ['time_step']} if op in ('ParseDate', 'ExtractDateTimeFeature') else dict(target_col='value', time_col='time_step', lags=[1])
            self.assertTrue(validate_step_params(op, params, ctx))

    def test_external_yaml_and_combined_candidate_rejected(self):
        for op in ('ExtractDateTimeFeature', 'CreateLagFeature'):
            p = Pipeline([PipelineStep(op, params={'cols': ['time_step']} if op == 'ExtractDateTimeFeature'
                else dict(target_col='value', time_col='time_step', lags=[1]))])
            assign_dag_structure(p, context())
            self.assertFalse(check_pipeline(Pipeline.from_yaml(p.to_yaml()), context(), require_mandatory=False).valid)

    def test_explicit_numeric_format_all_task_types(self):
        for task in ('rec', 'graph', 'tabular'):
            ctx = context(task, 'declared_fixture', {'time_step': {'kind': 'numeric_date', 'format': '%Y%m%d'}})
            p = build_default_params('ParseDate', ctx, random.Random(42))
            self.assertIsNotNone(p)
            self.assertEqual(validate_step_params('ParseDate', p, ctx), [])
            result = ParseDate(**p).transform(pd.DataFrame({'time_step': [20261103]}))
            self.assertEqual(result.time_step.dt.year.iloc[0], 2026)

    def test_epoch_requires_origin_and_output_format_is_not_input_format(self):
        ctx = context('tabular', 'fixture', {'time_step': {'kind': 'calendar', 'unit': 's'}})
        self.assertTrue(validate_step_params('ParseDate', {'cols': ['time_step'], 'target_format': '%Y%m%d'}, ctx))

    def test_known_rec_and_berka_modes_preserved(self):
        p, errors = resolve_time_params('ParseDate', {'cols': ['timestamp']}, 'kuairec')
        self.assertFalse(errors)
        self.assertEqual(p['time_rules']['timestamp'], {'unit': 's', 'origin': 'unix'})
        self.assertFalse(resolve_time_params('ParseDate', {'cols': ['date'], 'mode': 'int_yymmdd'}, 'berka')[1])
        self.assertFalse(resolve_time_params('ParseDate', {'cols': ['birth_number'], 'mode': 'berka_birth'}, 'berka')[1])

    def test_no_invention_by_yaml_or_column_name(self):
        for dataset in ('elliptic_bitcoin', 'unknown'):
            self.assertTrue(resolve_time_params('ExtractDateTimeFeature', {'cols': ['time_step'],
                'time_rules': {'time_step': {'unit': 's', 'origin': 'unix'}}}, dataset)[1])

    def test_mutation_cannot_override_declared_units(self):
        self.assertTrue(resolve_time_params('ParseDate', {'cols': ['timestamp'],
            'time_rules': {'timestamp': {'unit': 'ms', 'origin': 'unix'}}}, 'kuairec')[1])

    def test_datetime_declaration_and_missing_declaration(self):
        p = {'cols': ['time_step']}
        self.assertTrue(resolve_time_params('ParseDate', p, 'fixture')[1])
        self.assertFalse(resolve_time_params('ParseDate', p, 'fixture',
            {'time_step': {'kind': 'calendar', 'representation': 'datetime'}})[1])

    def test_rec_existing_history_contract_preserved(self):
        p, errors = resolve_time_params('CreateLagFeature', dict(target_col='value', time_col='timestamp', lags=[1]), 'kuairec', target='rating')
        self.assertFalse(errors)
        self.assertEqual(p['group_cols'], [])
        self.assertEqual(p['availability'], 'event_time')

    def test_rec_delayed_history_runtime(self):
        from dppbench.rec_protocol import history_transform
        op = CreateLagFeature('value', [1], group_cols=['entity'], time_col='timestamp', available_at_col='known')
        op._declared_history_rules = {c: {'unit': 's', 'origin': 'unix'} for c in ('timestamp', 'known')}
        frame = pd.DataFrame({'entity': ['a']*3, 'timestamp': [1,2,3], 'known': [1,9,3],
                              'value': [10.,20.,30.], '__split__': ['train','train','val']})
        self.assertEqual(history_transform(frame, op, 'timestamp').value_lag_1.iloc[-1], 10.)

    def history(self):
        return dict(entity_cols=['entity'], time_col='time_step', ordering='ascending',
                    same_time='exclude', mode='training_only', availability='event_time')

    def test_history_positive_and_negative_contracts(self):
        rules = {'time_step': {'kind': 'ordered_step'}}
        p = dict(target_col='value', time_col='time_step', lags=[1])
        self.assertFalse(resolve_time_params('CreateLagFeature', p, 'fixture', rules, self.history(), 'label')[1])
        for key, value in [('ordering', 'descending'), ('same_time', 'include'), ('mode', 'online'), ('availability', None)]:
            h = self.history(); h[key] = value
            self.assertTrue(resolve_time_params('CreateLagFeature', p, 'fixture', rules, h, 'label')[1])
        self.assertTrue(resolve_time_params('CreateLagFeature', dict(p, target_col='label'), 'fixture', rules, self.history(), 'label')[1])

    def test_history_execution_sorted_strict_train_only(self):
        cfg = {'feature': {'time_rules': {'time_step': {'kind': 'ordered_step'}}}, 'history_protocol': self.history()}
        d = SimpleNamespace(name='fixture', target_col='label', model_cfg=cfg)
        frame = pd.DataFrame({'entity': ['a']*5, 'time_step': [3, 1, 2, 2, 4],
                              'value': [30., 10., 20., 999., 40.], '__split__': ['train','train','train','val','val']})
        result = transform_partitioned_history(d, frame, CreateLagFeature('value', [1], time_col='time_step'))
        self.assertEqual(result.value_lag_1.iloc[[0,2,3,4]].tolist(), [20.,10.,10.,30.])

    def test_delayed_availability_excludes_unavailable_values(self):
        h = self.history(); h.pop('availability'); h['available_at_col'] = 'known'
        cfg = {'feature': {'time_rules': {c: {'kind': 'ordered_step'} for c in ('time_step','known')}}, 'history_protocol': h}
        frame = pd.DataFrame({'entity': ['a']*3, 'time_step': [1,2,3], 'known': [1,9,3],
                              'value': [10.,20.,30.], '__split__': ['train','train','val']})
        result = transform_partitioned_history(SimpleNamespace(name='fixture', target_col='label', model_cfg=cfg),
            frame, CreateLagFeature('value', [1], time_col='time_step'))
        self.assertEqual(result.value_lag_1.iloc[-1], 10.)


if __name__ == '__main__':
    unittest.main()
