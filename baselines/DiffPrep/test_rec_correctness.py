"""Frozen recommendation protocol and real-transform differentiable replay tests."""
import copy
import random
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch

from dppbench.dataset import RecData
from dppbench.rec_protocol import training_rows
from dppbench.models.inputs import build_feature_columns, df_to_input
from baselines.DiffPrep.rec_search import prepare_rec_search, RecTransformBank
from baselines.DiffPrep.slot_planner import make_slots
from baselines.DiffPrep.search_space import ContinuousPipeline
from baselines.DiffPrep.discretizer import discretize


def dataset(tmp_path):
    d = RecData('amazon_beauty')
    d.data_dir = str(tmp_path/'data')
    (tmp_path/'data').mkdir(exist_ok=True)
    std = tmp_path/'std_test'
    std.mkdir(exist_ok=True)
    d.col_types = {'user_id': 'categorical', 'item_id': 'categorical', 'item_brand': 'categorical',
                   'rating': 'categorical', 'timestamp': 'timestamp'}
    frame = pd.DataFrame({'user_id': ['u1', 'u2']*12, 'item_id': ['i1', 'i2', 'i3']*8,
                          'rating': [0, 1]*12, 'timestamp': range(24)})
    heldout = frame.iloc[:4].assign(timestamp=np.arange(100, 104))
    frame.to_parquet(std/'interaction_frozen.parquet', index=False)
    heldout.to_parquet(std/'std_test.parquet', index=False)
    d.item_df = pd.DataFrame({'item_id': ['i1', 'i2', 'i3'], 'item_brand': ['a', 'b', 'c']})
    d.interaction_df = pd.concat([frame, heldout], ignore_index=True)  # raw loader would see future rows
    cfg = {'feature': {'target_col': 'rating', 'seq_cols': {'item_id_seq': 'item_id'}, 'seq_max_len': 20},
           'train': {'val_ratio': .25, 'seed': 42}, 'model_params': {'embedding_dim': 4, 'task': 'binary'}}
    return d, cfg, std


def setup_search(tmp_path, flex=False):
    d, cfg, std = dataset(tmp_path)
    ctx, structural, audit = prepare_rec_search(d, cfg, 42, 10)
    ctx.data_name = 'amazon_beauty'
    slots = make_slots('rec', ctx, max_pipeline_len=10)
    bank = RecTransformBank(d, ctx, slots, 42, flex=flex)
    cont = ContinuousPipeline(slots, n_features=len(bank.feature_names), feature_names=bank.feature_names, flex=flex)
    cont.rec_bank, cont.rec_context, cont.rec_seed, cont.rec_data = bank, ctx, 42, d
    cont.rec_sequence_step = d._diffprep_sequence_step
    cont.register_buffer('rec_candidate_mask', torch.tensor(bank.valid_mask))
    return d, cfg, ctx, structural, audit, bank, cont


def test_search_reads_only_frozen_train_and_projected_test_identities(tmp_path):
    d, cfg, std = dataset(tmp_path)
    read = pd.read_parquet
    calls = []
    def audited(path, *args, **kwargs):
        calls.append((str(path), kwargs.get('columns')))
        if str(path).endswith('std_test.parquet'):
            assert kwargs['columns'] == ['user_id', 'item_id', 'timestamp']
        return read(path, *args, **kwargs)
    with patch.object(pd, 'read_parquet', side_effect=audited):
        ctx, steps, audit = prepare_rec_search(d, cfg, 42, 10)
    assert len(calls) == 2
    assert d.interaction_df.timestamp.max() < 100
    assert len(training_rows(d.interaction_df)) == 10
    assert len(d.interaction_df.loc[d.interaction_df['__split__'].eq('val')]) == 6
    assert 'item_brand' in d.interaction_df
    assert steps[0].op == 'JoinTable'
    assert audit['identity_columns'] == ['user_id', 'item_id', 'timestamp']


def test_search_refuses_overlap_before_optimization(tmp_path):
    d, cfg, std = dataset(tmp_path)
    pd.read_parquet(std/'interaction_frozen.parquet').iloc[:2].to_parquet(std/'std_test.parquet', index=False)
    with pytest.raises(ValueError, match='overlap'):
        prepare_rec_search(d, cfg, 42, 10)


def test_actual_brand_onehot_matches_export_and_din_inputs(tmp_path):
    d, cfg, ctx, structural, audit, bank, cont = setup_search(tmp_path)
    assert bank.feature_names == ['item_brand']
    slot = next(i for i, s in enumerate(cont.slots) if 'OneHotEncode' in s.candidates)
    choice = cont.slots[slot].candidates.index('OneHotEncode')
    with torch.no_grad():
        cont.tau[slot].fill_(-100)
        cont.tau[slot, :, choice] = 100
    rows = torch.arange(len(bank.frame)).float().reshape(-1, 1)
    search = cont(rows, torch.zeros_like(rows)).detach().numpy()
    pipe = discretize(cont, cont.slots, ctx, rng=random.Random(42), structural_steps=structural)
    assert [s.op for s in pipe.steps] == ['JoinTable', 'OneHotEncode', 'CreateSequence']
    assert pipe.steps[1].params['cols'] == ['item_brand']
    # Replay on the same allowed rows so statistics/columns must match exactly.
    replay = copy.copy(d)
    replay.col_types = {'user_id': 'categorical', 'item_id': 'categorical', 'item_brand': 'categorical',
                        'rating': 'categorical', 'timestamp': 'timestamp', 'item_id_seq': 'categorical_list'}
    replay.model_cfg = copy.deepcopy(cfg)
    replay.interaction_df = bank.frame.copy()
    op = replay._load_op('OneHotEncode')(**pipe.steps[1].params)
    replay.interaction_df = replay._transform_rec_dag_df(replay.interaction_df, op, op.name)
    replay._remap_ids()
    fc = build_feature_columns(training_rows(replay.interaction_df), cfg, replay.col_types)
    x, _ = df_to_input(replay.interaction_df, fc, cfg)
    for column in op.output_col_types:
        position = bank.union_columns[0].index(column)
        np.testing.assert_allclose(search[:, position], x[column])
    assert all(c.startswith('item_brand_') for c in op.output_col_types)


def test_architecture_gradients_and_finite_difference_bilevel_update_survive(tmp_path):
    from baselines.DiffPrep.surrogate import RecSurrogate
    from baselines.DiffPrep.trainer import DiffPrepTrainer
    d, cfg, ctx, structural, audit, bank, cont = setup_search(tmp_path)
    model = RecSurrogate(3, 4, feat_dim=bank.output_dim, emb_dim=4, hidden=8)
    rows = torch.arange(len(bank.frame)).float().reshape(-1, 1)
    y = torch.tensor(bank.frame.rating.to_numpy(), dtype=torch.float32)
    ids = torch.zeros(len(rows), dtype=torch.long)
    tr = np.flatnonzero(bank.frame['__split__'].eq('train'))
    va = np.flatnonzero(bank.frame['__split__'].eq('val'))
    def loss(c, m, indices):
        out = c(rows[indices], torch.zeros_like(rows[indices]))
        return torch.nn.functional.binary_cross_entropy_with_logits(m(ids[indices], ids[indices], out), y[indices])
    trainer = DiffPrepTrainer(cont, model, lambda c,m: loss(c,m,tr),
        lambda c,m: (loss(c,m,va), 0.), lambda c: [], n_epochs=2, second_order=True, verbose=False)
    before = cont.tau.detach().clone()
    history = trainer.train()
    assert len(history) == 2
    assert torch.isfinite(cont.tau).all()
    assert not torch.equal(before, cont.tau.detach())


def test_heldout_labels_cannot_change_search_transform_bank(tmp_path):
    d, cfg, std = dataset(tmp_path)
    ctx, steps, _ = prepare_rec_search(d, cfg, 42, 10)
    slots = make_slots('rec', ctx)
    a = RecTransformBank(d, ctx, slots, 42)
    changed = copy.copy(d)
    changed.interaction_df = d.interaction_df.copy()
    mask = changed.interaction_df['__split__'].ne('train')
    changed.interaction_df.loc[mask, 'rating'] = 1-changed.interaction_df.loc[mask, 'rating']
    b = RecTransformBank(changed, ctx, slots, 42)
    for key in a.blocks:
        np.testing.assert_allclose(a.blocks[key].toarray(), b.blocks[key].toarray())


def fake_loader(source):
    def load(data):
        data.col_types = dict(source.col_types)
        data.interaction_df = source.interaction_df.copy(deep=True)
        data.item_df = source.item_df.copy(deep=True)
        return data.interaction_df, data.item_df
    return load


def test_two_epoch_diffprep_orchestration_uses_verified_protocol(tmp_path):
    from baselines.DiffPrep.diffprep import DiffPrep
    from dppbench.tasks.amazon_beauty.amazon_beauty_data import AmazonBeautyData
    d, cfg, _ = dataset(tmp_path)
    root = __import__('pathlib').Path(__file__).resolve().parents[2]
    with patch.object(AmazonBeautyData, 'load_data', fake_loader(d)):
        runner = DiffPrep(str(root/'dppbench/tasks/amazon_beauty'), 'amazon_beauty',
                          data_dir=d.data_dir, output_dir=str(tmp_path/'output'),
                          n_epochs=2, small_n=10, batch_size=4, eval_full=False,
                          verbose=False, device='cpu')
        result = runner.run()
    assert len(result['search_history']) == 2
    contract = result['search_contract']
    assert contract['version'] == 5
    assert contract['search_feature_columns'] == ['item_brand']
    assert contract['search_joins'] == contract['exported_joins']
    assert contract['representation']['adaptation'] == 'exact_real_transform_path_mixture_v1'


def test_common_hpo_and_final_executor_schema_equal_on_unseen_test_values(tmp_path):
    from baselines.common.executor import TrainingExecutor
    from baselines.common.pipeline import Pipeline, PipelineStep, assign_dag_structure
    from dppbench.tasks.amazon_beauty.amazon_beauty_data import AmazonBeautyData
    d, cfg, directory = dataset(tmp_path)
    heldout = pd.read_parquet(directory/'std_test.parquet')
    heldout['user_id'], heldout['item_id'] = 'unseen-user', 'unseen-item'
    heldout.to_parquet(directory/'std_test.parquet', index=False)
    root = __import__('pathlib').Path(__file__).resolve().parents[2]
    ctx = __import__('baselines.common.data_ctx', fromlist=['infer_data_context']).infer_data_context('amazon_beauty', d, 'rec')
    pipe = Pipeline([PipelineStep('JoinTable', params={'method': 'rec', 'item_df': '$item_df',
                      'user_col': 'user_id', 'item_col': 'item_id', 'how': 'left'}),
                     PipelineStep('CreateSequence', params={'max_len': 20})])
    assign_dag_structure(pipe, ctx, random.Random(42))
    config = {'train': {'epochs': 1, 'early_stopping_patience': 0, 'batch_size': 8},
              'model_params': {'embedding_dim': 4, 'dnn_hidden_units': [8, 4], 'att_hidden_size': [4, 2]}}
    results = []
    from dppbench.models.base_model import BaseModel
    original_evaluate = BaseModel.evaluate
    with patch.object(AmazonBeautyData, 'load_data', fake_loader(d)):
        for final in (False, True):
            executor = TrainingExecutor(str(root/'dppbench/tasks/amazon_beauty'), 'amazon_beauty',
                                        data_dir=d.data_dir, model_name='DIN', model_config=config, seed=42)
            try:
                executor.set_evaluate_std_test(final)
                executor.write_yaml(pipe.to_yaml())
                test_calls = []
                def counted(model, x, y, *args, **kwargs):
                    if len(y) == len(heldout):
                        test_calls.append(len(y))
                    return original_evaluate(model, x, y, *args, **kwargs)
                with patch.object(BaseModel, 'evaluate', counted):
                    result = executor.run_training()
                assert len(test_calls) == int(final), 'HPO must test zero times; final exactly once'
                assert result['success'], result['error']
                results.append(result)
            finally:
                executor.cleanup()
    assert results[0]['feature_schema_sha256'] == results[1]['feature_schema_sha256']
    assert 'std_test_auc' not in results[0]['metrics']
    assert results[1]['metrics']['std_test_n_rows'] == 4
    # Final replay must fail before fitting if its schema differs from HPO.
    with patch.object(AmazonBeautyData, 'load_data', fake_loader(d)):
        executor = TrainingExecutor(str(root/'dppbench/tasks/amazon_beauty'), 'amazon_beauty',
                                    data_dir=d.data_dir, model_name='DIN', model_config=config, seed=42)
        try:
            executor.expected_rec_schema_sha256 = 'deliberately-incompatible-hpo-schema'
            executor.write_yaml(pipe.to_yaml())
            failure = executor.run_training()
            assert not failure['success'] and 'schema mismatch' in failure['error']
        finally:
            executor.cleanup()


def test_numeric_and_categorical_candidates_and_flex_are_not_removed(tmp_path):
    d, cfg, _ = dataset(tmp_path)
    d.item_df['price'] = [1., 2., 3.]
    d.col_types['price'] = 'numeric'
    ctx, structural, _ = prepare_rec_search(d, cfg, 42, 10)
    slots = make_slots('rec', ctx)
    bank = RecTransformBank(d, ctx, slots, 42, flex=True)
    assert set(bank.feature_names) == {'item_brand', 'price'}
    cont = ContinuousPipeline(slots, flex=True, n_features=2, feature_names=bank.feature_names)
    cont.rec_bank = bank
    cont.register_buffer('rec_candidate_mask', torch.tensor(bank.valid_mask))
    # Applicable operator families stay present; only invalid column/operator
    # pairs are masked (scaling strings would fail the deployed real operator).
    assert any('OneHotEncode' in s.candidates for s in slots)
    assert any('ScaleFeature' in s.candidates for s in slots)
    rows = torch.arange(len(bank.frame)).float().reshape(-1, 1)
    value = cont(rows, torch.zeros_like(rows))
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert cont.tau.grad is not None


def test_flex_retains_learned_order_gradients_for_real_compositions(tmp_path):
    from baselines.DiffPrep.slot_planner import Slot
    from baselines.common.op_spec import OpCategory
    d, cfg, _ = dataset(tmp_path)
    d.item_df['price'] = [1., 2., 4.]
    d.col_types['price'], d.col_types['item_brand'] = 'numeric', 'text'
    ctx, _, _ = prepare_rec_search(d, cfg, 42, 10)
    slots = [Slot(0, OpCategory.PREPROCESSING, ['TransformPower', 'Identity'], 'soft'),
             Slot(1, OpCategory.PREPROCESSING, ['ScaleFeature', 'Identity'], 'soft')]
    bank = RecTransformBank(d, ctx, slots, 42, flex=True)
    cont = ContinuousPipeline(slots, flex=True, n_features=1, feature_names=bank.feature_names)
    cont.rec_bank = bank
    cont.register_buffer('rec_candidate_mask', torch.tensor(bank.valid_mask))
    rows = torch.arange(len(bank.frame)).float().reshape(-1, 1)
    value = cont(rows, torch.zeros_like(rows))
    value.square().sum().backward()
    assert cont.theta is not None and cont.theta.grad is not None
    assert torch.isfinite(cont.theta.grad).all()
    assert cont.theta.grad.abs().sum() > 0


def test_candidate_remapping_does_not_mutate_source_side_tables(tmp_path):
    d, cfg, _ = dataset(tmp_path)
    ctx, _, _ = prepare_rec_search(d, cfg, 42, 10)
    before = d.item_df.copy(deep=True)
    RecTransformBank(d, ctx, make_slots('rec', ctx), 42)
    pd.testing.assert_frame_equal(d.item_df, before)
