import unittest

import numpy as np
import pandas as pd
import torch

from baselines.DAGPipe.categorical import ENCODERS, FrozenCategoryBranch, fit_category_state
from baselines.DAGPipe.pipeline import DAGPipePipeline
from baselines.DAGPipe.runtime import InputEncoder, limited_layers, make_model, reference_operation
from baselines.DAGPipe.vendor import operations
from dppbench.operators.preprocessing.scale_feature import ScaleFeature
from dppbench.operators.preprocessing.target_encode import TargetEncode
from dppbench.operators.feature_engineering.frequency_encode import FrequencyEncode


class PublicSemanticsTests(unittest.TestCase):
    def test_elliptic_historical_layout_counts_bypass_nodes(self):
        layers = limited_layers(9, operator_space='legacy_daghie', encoder_bypasses=True)
        self.assertEqual(layers, [['MeanImputer', 'MedianImputer', 'KNNImputer'],
                                  ['Identity', 'Identity'], ['StandardScaler', 'RobustScaler'],
                                  ['StandardScaler', 'RobustScaler']])
        self.assertEqual(1 + sum(map(len, layers)), 10)

    def test_legacy_space_budget_and_missing_value_replay(self):
        expected = [['MeanImputer', 'MedianImputer', 'KNNImputer'], list(ENCODERS),
                    ['StandardScaler', 'RobustScaler'], ['StandardScaler', 'RobustScaler']]
        self.assertEqual(limited_layers(9, True, 'legacy_daghie'), expected)
        train = pd.DataFrame({'cat': ['a', 'b', 'a'], 'x': [1., 3., None]})
        encoder = InputEncoder(preserve_missing=True).fit(train)
        state = fit_category_state(train, pd.Series([0., 1., 1.]), encoder.state, ENCODERS)
        model = make_model(expected, 2, specs=encoder.state, category_state=state,
                           scale_protocol='public_v1', operator_space='legacy_daghie')
        ref = encoder.transform(train)
        self.assertTrue(np.isnan(ref[-1, 1]))
        pipeline = DAGPipePipeline(dict(format='dppbench.DAGPipe.reference.v3', task='fixture', seed=42,
            encoder=encoder.state, layers=expected, category_state=state, scale_protocol='public_v1',
            operator_space='legacy_daghie', state_dict={k: v.tolist() for k, v in model.state_dict().items()},
            reference=ref.tolist()))
        self.assertEqual(len(pipeline), 10)
        replay = DAGPipePipeline.from_yaml(pipeline.to_yaml())
        output = replay.transform(train)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output[-1:], replay.transform(train.iloc[-1:]), rtol=1e-10)

    def test_budget_and_categorical_eligibility(self):
        for slots in range(1, 10):
            layers = limited_layers(slots, categorical=True)
            self.assertEqual(sum(map(len, layers)), slots)
            if slots >= 3:
                self.assertTrue(set(ENCODERS).issubset(layers[0]))
        self.assertFalse(set(ENCODERS) & set(sum(limited_layers(9), [])))

    def test_scalers_match_public_finite_input(self):
        train = pd.DataFrame({'a': [1., 2., 3., 7.], 'constant': [2.]*4,
                              'tiny': [0., 1e-9, 2e-9, 3e-9]})
        query = pd.DataFrame({'a': [8., 1.], 'constant': [5., 2.], 'tiny': [4e-9, 0.]})
        x = torch.tensor(pd.concat([train, query]).to_numpy())
        for method, cls in [('standard', operations.StandardScaler), ('robust', operations.RobustScaler),
                            ('minmax', operations.MinMaxScaler)]:
            public = ScaleFeature(cols=list(train), method=method)
            public.transform(train)
            actual = reference_operation(cls(), x, len(train), 'public_v1')[len(train):]
            np.testing.assert_allclose(actual.numpy(), public.transform(query).to_numpy(), rtol=1e-6, atol=1e-7)

    def test_encoders_public_values_and_checkpoint_replay(self):
        train = pd.DataFrame({'category': ['a', 'b', 'a', None], 'numeric': [1., 2., 3., 4.]})
        labels = pd.Series([0., 0., 1., 1.])
        query = pd.DataFrame({'category': ['a', 'b', 'unknown', None], 'numeric': [4., 3., 2., 1.]})
        encoder = InputEncoder().fit(train)
        state = fit_category_state(train, labels, encoder.state, ENCODERS)
        x = torch.tensor(encoder.transform(query))
        target = TargetEncode(cols=['category'], target_col='y')
        target.transform(train.assign(y=labels))
        frequency = FrequencyEncode(cols=['category'])
        frequency.transform(train)
        for name, expected in [('TargetEncoder', target.transform(query)['category_y']),
                               ('FrequencyEncoder', frequency.transform(query)['category_freq'])]:
            actual = FrozenCategoryBranch(encoder.state, state[name])(x)
            np.testing.assert_allclose(actual[:, 0], expected.to_numpy())
            torch.testing.assert_close(actual[:, 1], x[:, 1])
        layers = limited_layers(9, categorical=True)
        model = make_model(layers, 2, specs=encoder.state, category_state=state, scale_protocol='public_v1')
        reference = torch.tensor(encoder.transform(train))
        model(torch.cat([reference, x]), reference_rows=len(reference)).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        pipeline = DAGPipePipeline(dict(format='dppbench.DAGPipe.reference.v3', task='fixture', seed=42,
            encoder=encoder.state, layers=layers, category_state=state, scale_protocol='public_v1',
            state_dict={k: v.tolist() for k, v in model.state_dict().items()}, reference=reference.tolist()))
        self.assertEqual(len(pipeline), 10)
        replay = DAGPipePipeline.from_yaml(pipeline.to_yaml())
        np.testing.assert_allclose(replay.transform(query), pipeline.transform(query), rtol=1e-10)
        np.testing.assert_allclose(replay.transform(query.iloc[:1]), pipeline.transform(query)[:1], rtol=1e-10)
