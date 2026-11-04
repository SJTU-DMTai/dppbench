import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from .runtime import InputEncoder, make_model, reference_forward, FrozenTransform
from .pipeline import DAGPipePipeline
from .protocol_v4 import (SPACE, allocate_layers, fit_tables, row_folds, TypedOperation,
                         validate_raw_features, fit_input_encoder, freeze_statistics)


def fixture():
    frame = pd.DataFrame({'n': np.arange(24, dtype=float), 'c': ['a', 'b', 'c'] * 8})
    frame.loc[2, 'n'] = np.nan
    labels = pd.Series(np.arange(24) % 2, dtype=float)
    encoder = fit_input_encoder(frame)
    values = encoder.transform(frame)
    layers = allocate_layers(encoder.state, values, 9)
    names = [n for layer in layers for n in layer if n in ('TargetEncoder', 'FrequencyEncoder')]
    state = fit_tables(frame, labels, encoder.state, values, names, 42)
    model = make_model(layers, 2, specs=encoder.state, category_state=state,
                       scale_protocol='public_v1', operator_space=SPACE)
    payload = dict(format='dppbench.DAGPipe.reference.v4', task='synthetic', seed=42,
                   encoder=encoder.state, reference=values.tolist(), layers=layers,
                   category_state=state, operator_space=SPACE, scale_protocol='public_v1',
                   fit_protocol='all_legal_training_rows', training_rows=len(values), max_pipeline_len=10,
                   allow_target_encoding=True,
                   frozen_nodes=freeze_statistics(copy.deepcopy(model), torch.tensor(values, dtype=torch.float64)),
                   dtype='float64', transform_batch_size=7,
                   state_dict={k: v.tolist() for k, v in model.state_dict().items()})
    return frame, labels, encoder, values, model, payload


class ProtocolV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_budget_and_typed_allocation(self):
        frame, _, encoder, values, _, payload = fixture()
        self.assertEqual(len(DAGPipePipeline(payload)), 10)
        self.assertIn('TargetEncoder', sum(payload['layers'], []))
        self.assertIn('FrequencyEncoder', sum(payload['layers'], []))
        temporal_layers = allocate_layers(encoder.state, values, 9, allow_target=False, allow_frequency=False)
        self.assertNotIn('TargetEncoder', sum(temporal_layers, []))
        self.assertNotIn('FrequencyEncoder', sum(temporal_layers, []))
        for budget in range(4, 10):
            self.assertLessEqual(sum(map(len, allocate_layers(encoder.state, values, budget))), budget)
        with self.assertRaisesRegex(ValueError, 'budget'):
            allocate_layers(encoder.state, values, 3)
        numeric = InputEncoder(preserve_missing=True).fit(frame[['n']])
        self.assertNotIn('TargetEncoder', sum(allocate_layers(numeric.state, numeric.transform(frame), 9), []))
        broken = copy.deepcopy(payload)
        broken['reference'] = broken['reference'][:-1]
        with self.assertRaisesRegex(ValueError, 'row count'):
            DAGPipePipeline(broken)
        broken = copy.deepcopy(payload)
        broken['layers'][0].append('MeanImputer')
        with self.assertRaises(ValueError):
            DAGPipePipeline(broken)

    def test_own_label_and_same_fold_labels_cannot_affect_encoded_row(self):
        frame, labels, encoder, values, _, payload = fixture()
        folds = row_folds(values, 5, 42)
        altered = labels.copy()
        altered[folds == folds[0]] += 1000
        new_state = fit_tables(frame, altered, encoder.state, values,
                               ['TargetEncoder', 'FrequencyEncoder'], 42)
        source = torch.tensor(values, dtype=torch.float64)
        original = TypedOperation('TargetEncoder', encoder.state, payload['category_state'])
        changed = TypedOperation('TargetEncoder', encoder.state, new_state)
        a = original.op(source.nan_to_num(), source)
        b = changed.op(source.nan_to_num(), source)
        torch.testing.assert_close(a[folds == folds[0]], b[folds == folds[0]], rtol=0, atol=0)
        # Check entire DAG: numeric scaler statistics must not leak other-fold
        # target tables back into this row's categorical channel.
        model = make_model(payload['layers'], 2, specs=encoder.state, category_state=new_state,
                           scale_protocol='public_v1', operator_space=SPACE)
        original_model = FrozenTransform(payload).model
        torch.testing.assert_close(reference_forward(model, source[:1], source),
                                   reference_forward(original_model, source[:1], source), rtol=1e-12, atol=1e-12)

    def test_checkpoint_batch_permutation_and_unknown_categories(self):
        frame, _, _, values, search_model, payload = fixture()
        query = frame.iloc[:8].copy()
        query.loc[0, 'c'] = 'unseen'
        pipeline = DAGPipePipeline(payload)
        before = pipeline.to_yaml()
        result = pipeline.transform(query)
        loaded = DAGPipePipeline.from_yaml(before)
        np.testing.assert_allclose(result, loaded.transform(query), rtol=0, atol=0)
        np.testing.assert_allclose(result, np.concatenate([loaded.transform(query.iloc[i:i+1]) for i in range(len(query))]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[::-1], loaded.transform(query.iloc[::-1]), rtol=1e-12, atol=1e-12)
        self.assertEqual(before, pipeline.to_yaml())
        frozen = FrozenTransform(payload)
        encoded = torch.tensor(frozen.encoder.transform(query), dtype=torch.float64)
        baseline = reference_forward(frozen.model, encoded, frozen.reference)
        reference = torch.tensor(values, dtype=torch.float64)
        torch.testing.assert_close(reference_forward(search_model, encoded, reference),
                                   baseline, rtol=1e-12, atol=1e-12)
        # Deployment must use serialized stats rather than refitting on rows.
        frozen.reference.fill_(float('nan'))
        torch.testing.assert_close(baseline, reference_forward(frozen.model, encoded, frozen.reference), rtol=0, atol=0)

    def test_public_numeric_values_and_gradients(self):
        from dppbench.operators.preprocessing.scale_feature import ScaleFeature
        from dppbench.operators.cleaning.handle_m_v import HandleMV
        specs = [{'column': 'n', 'kind': 'numeric'}]
        source = torch.tensor([[0.], [1e-9], [2e-9], [3e-9], [10.]], dtype=torch.float64)
        route = torch.tensor([[.3]], dtype=torch.float64, requires_grad=True)
        for name, method in [('StandardScaler', 'standard'), ('RobustScaler', 'robust'), ('MinMaxScaler', 'minmax')]:
            node = TypedOperation(name, specs, {})
            weighted = source * route
            out = node.execute(weighted, 4, source, source, route)
            public = ScaleFeature(cols=['n'], method=method)
            public.transform(pd.DataFrame(weighted[:4].detach().numpy(), columns=['n']))
            expected = public.transform(pd.DataFrame(weighted[4:].detach().numpy(), columns=['n'])).to_numpy()
            np.testing.assert_allclose(out[4:].detach(), expected, rtol=2e-6)
        missing = torch.tensor([[0.], [2.], [float('nan')], [10.]], dtype=torch.float64)
        for name, method in [('MeanImputer', 'mean'), ('MedianImputer', 'median')]:
            node = TypedOperation(name, specs, {})
            out = node.execute(missing.nan_to_num() * route, 3, missing, missing.nan_to_num(), route)
            public = HandleMV(cols=['n'], method=method)
            public.transform(pd.DataFrame(missing[:3].numpy(), columns=['n']))
            expected = public.transform(pd.DataFrame(missing.numpy(), columns=['n'])).to_numpy() * .3
            np.testing.assert_allclose(out.detach(), expected)
        _, _, _, values, model, _ = fixture()
        ref = torch.tensor(values, dtype=torch.float64)
        loss = reference_forward(model, ref[:5], ref).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.routers[0].logits.grad.abs().sum().item(), 0.)
        self.assertGreater(model.layer_adj_logits.grad.abs().sum().item(), 0.)

    def test_invalid_inputs_fail_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            validate_raw_features(pd.DataFrame({'n': [np.inf]}), [{'column': 'n', 'kind': 'numeric'}])
        with self.assertRaisesRegex(ValueError, 'all-missing'):
            allocate_layers([{'kind': 'numeric'}], np.array([[np.nan], [np.nan]]), 9)
        encoded = fit_input_encoder(pd.DataFrame({'n': ['1.5', '2.5']}), column_types={'n': 'numeric'})
        self.assertEqual(encoded.state[0]['kind'], 'numeric')
        np.testing.assert_array_equal(encoded.transform(pd.DataFrame({'n': ['1.5']})), [[1.5]])
        with self.assertRaisesRegex(ValueError, 'dataset protocol'):
            fit_input_encoder(pd.DataFrame({'time': ['2020-01-01']}), column_types={'time': 'timestamp'})

    def test_real_runner_search_and_replay_without_formal_experiment(self):
        from .dagpipe import DAGPipe
        from baselines.common.search_cost import SearchCost, search_cost_scope
        from baselines.common.budget import Budget, budget_scope
        frame, labels, _, _, _, _ = fixture()
        frame['label'] = labels
        data = SimpleNamespace(target_col='label', id_col=None, col_types={'c': 'categorical'})
        executor = SimpleNamespace(task_type='tabular', get_model_config=lambda: {'model_params': {'task': 'binary'}},
                                   load_data=lambda: data, cleanup=lambda: None)
        with tempfile.TemporaryDirectory() as folder, patch('baselines.DAGPipe.dagpipe.TrainingExecutor', return_value=executor), \
             patch('baselines.DAGPipe.dagpipe.training_splits', return_value={'train': frame, 'val': frame.iloc[:6].copy()}):
            runner = DAGPipe(task_dir=folder, data_name='synthetic', output_dir=folder, verbose=False)
            runner.cfg.update(epochs=2, batch_size=8)
            cost = SearchCost('DAGPipe')
            budget = Budget(max_pipeline_length=10, max_evaluations=1)
            with search_cost_scope(cost), budget_scope(budget):
                result = runner.run()
            self.assertEqual(budget.evaluations_used, 0)
            operations = cost.snapshot()['operations']
            self.assertEqual(operations['search/optimizer.joint']['completed'], 6)
            self.assertEqual(operations['search/proxy_work.train']['forwarded_rows'], 6 * (16+8))
            self.assertEqual(operations['search/operator_fit.final_statistics']['completed'], 1)
            payload = runner.best_pipeline.payload
            self.assertEqual(payload['training_rows'], 24)
            self.assertEqual(len(payload['reference']), 16)
            self.assertEqual(payload['format'], 'dppbench.DAGPipe.reference.v4')
            loaded = DAGPipePipeline.from_yaml(result['best_pipeline_yaml'])
            np.testing.assert_allclose(loaded.transform(frame), runner.best_pipeline.transform(frame), rtol=0, atol=0)
            self.assertTrue((Path(folder) / 'DAGPipe_search.json').exists())
            repeat = DAGPipe(task_dir=folder, data_name='synthetic', output_dir=Path(folder)/'repeat', verbose=False)
            repeat.cfg.update(epochs=2, batch_size=8)
            self.assertEqual(result['best_pipeline_yaml'], repeat.run()['best_pipeline_yaml'])

    def test_categorical_only_temporal_allocation_has_real_choices(self):
        frame = pd.DataFrame({'c': ['a', 'b', 'a', 'b']})
        encoder = fit_input_encoder(frame)
        layers = allocate_layers(encoder.state, encoder.transform(frame), 9, allow_target=False, allow_frequency=False)
        self.assertEqual(layers, [['Identity']])

    def test_graph_runner_uses_training_mask_only(self):
        from .dagpipe import DAGPipe
        frame = pd.DataFrame({'feat_0': np.arange(12, dtype=float), 'label': [0, 1] * 6})
        data = SimpleNamespace(target_col='label', id_col='txId', col_types={'feat_0': 'numeric'})
        executor = SimpleNamespace(task_type='graph', get_model_config=lambda: {'model_params': {'task': 'binary'}},
                                   load_data=lambda: data, cleanup=lambda: None)
        with tempfile.TemporaryDirectory() as folder, patch('baselines.DAGPipe.dagpipe.TrainingExecutor', return_value=executor), \
             patch('baselines.DAGPipe.dagpipe.graph_splits', return_value={'train': frame.iloc[:8], 'val': frame.iloc[8:]}):
            runner = DAGPipe(task_dir=folder, data_name='synthetic_graph', output_dir=folder, verbose=False)
            runner.cfg.update(epochs=1, batch_size=4)
            runner.run()
            self.assertEqual(runner.best_pipeline.payload['training_rows'], 8)
            self.assertEqual(np.asarray(runner.best_pipeline.payload['reference']).max(), 7.)
            self.assertFalse(runner.best_pipeline.payload['allow_frequency_encoding'])


if __name__ == '__main__':
    unittest.main()
