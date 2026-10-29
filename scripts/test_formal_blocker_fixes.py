"""Small analytic/stochastic/real-model regressions; never run full experiments."""
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from baselines.DiffPrep.rec_search import prepare_rec_search, fixed_rec_list_inputs
from baselines.DiffPrep.surrogate import RecSurrogate
from baselines.DiffPrep.trainer import DiffPrepTrainer
from baselines.DiffPrep.test_rec_correctness import dataset
from dppbench.models.inputs import build_feature_columns, df_to_input
from dppbench.models.sim import SIM
from dppbench.rec_protocol import training_rows


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_hvp_matches_autograd_with_same_batch_dropout_and_rng(device):
    class Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tau = torch.nn.Parameter(torch.tensor(.3, device=device, dtype=torch.float64))
            self.theta = None
    c = Architecture()
    m = torch.nn.Linear(1, 1, bias=False, device=device, dtype=torch.float64)
    with torch.no_grad():
        m.weight.fill_(.7)
    draws = []
    data = torch.linspace(.1, 2., 20, dtype=torch.float64, device=device).reshape(-1, 1)
    def loss(a, model):
        ids = torch.randperm(len(data), device=device)[:8]
        x = torch.nn.functional.dropout(data[ids], p=.3, training=True)
        py, num = random.random(), float(np.random.rand())
        draws.append((ids.detach().cpu().tolist(), x.detach().cpu().clone(), py, num))
        return ((model(x) - a.tau*x - .2*(py+num))**2).mean()
    trainer = DiffPrepTrainer(c, m, loss, lambda a,b: (loss(a,b), 0.), lambda a: [],
                              n_epochs=1, eps_finite_diff=1e-4, second_order=True, verbose=False)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    backup = [p.detach().clone() for p in m.parameters()]
    numerical = trainer._hessian_vector_product(backup, [torch.ones_like(m.weight)])[0]
    assert draws[0][0] == draws[1][0]
    assert torch.equal(draws[0][1], draws[1][1])
    assert draws[0][2:] == draws[1][2:]
    # Autograd for exactly the same sampled inputs and dropout mask.
    x = draws[0][1].to(device)
    exact_loss = ((m(x)-c.tau*x-.2*sum(draws[0][2:]))**2).mean()
    dw = torch.autograd.grad(exact_loss, m.weight, create_graph=True)[0]
    exact = torch.autograd.grad(dw.sum(), c.tau)[0]
    torch.testing.assert_close(numerical, exact, atol=1e-8, rtol=1e-7)
    assert torch.isfinite(numerical) and numerical.abs() > 0
    torch.testing.assert_close(m.weight, backup[0])
    # Random streams advance once, not twice, and a repeated HVP changes draws.
    trainer._hessian_vector_product(backup, [torch.ones_like(m.weight)])
    assert draws[2][0] != draws[0][0]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('k', [2, 5, 8])
def test_sim_recent_valid_suffix_and_backward(device, k):
    lengths = torch.tensor([0, 1, 2, 3, 5], device=device)
    keys = torch.arange(5*5*2, device=device, dtype=torch.float32).reshape(5, 5, 2).requires_grad_()
    output, out_lengths = SIM._recent_top_k(type('Stub', (), {'sim_top_k': k})(), keys, lengths)
    assert output.shape == (5, min(k, 5), 2)
    for row, length in enumerate(lengths.tolist()):
        n = min(length, k)
        assert int(out_lengths[row]) == n
        torch.testing.assert_close(output[row, :n], keys[row, max(0, length-k):length])
        assert torch.count_nonzero(output[row, n:]) == 0
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(keys.grad).all()
    mask = torch.arange(5, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    assert torch.count_nonzero(keys.grad[~mask]) == 0


@pytest.mark.parametrize('device', DEVICES)
def test_sim_tiny_forward_backward_oov_and_padding(device):
    from dppbench.models.inputs import SparseFeat, VarLenSparseFeat
    cols = [SparseFeat('user_id', 3, 4), SparseFeat('item_id', 5, 4),
            VarLenSparseFeat(SparseFeat('item_id_seq', 5, 4, embedding_name='item_id'),
                             5, length_name='item_id_seq_length', is_attention=True)]
    model = SIM(cols, sim_top_k=2, dnn_hidden_units=(8, 4), att_hidden_size=(4, 2), device=device)
    # Valid first token 0 is an OOV; a zero-length row has only padding.
    x = torch.tensor([[1, 2, 0, 3, 0, 0, 0, 2], [1, 2, 0, 0, 0, 0, 0, 0],
                      [1, 2, 1, 2, 3, 4, 1, 5]], device=device, dtype=torch.float32)
    out = model(x)
    assert torch.isfinite(out).all()
    torch.nn.functional.binary_cross_entropy(out.flatten(), torch.tensor([1., 0., 1.], device=device)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.embedding_dict['item_id'].weight.grad[0].abs().sum() > 0


@pytest.mark.parametrize('name', ['item_genres', 'item_categories', 'user_friend_list'])
def test_fixed_lists_match_din_and_ignore_heldout_labels(tmp_path, name):
    d, cfg, _ = dataset(tmp_path)
    d.col_types[name] = 'categorical_list'
    if name == 'user_friend_list':
        d._user_id_related_cols = [name]
        d.user_df = pd.DataFrame({'user_id': ['u1', 'u2'], name: [['u2'], ['u1']]})
    else:
        d.item_df[name] = [['action'], ['drama'], ['action', 'drama']]
    prepare_rec_search(d, cfg, 42, 10)
    # Ensure a held-out-only token has no representation-fitting influence.
    val_pos = d.interaction_df.index[d.interaction_df['__split__'].eq('val')][0]
    d.interaction_df.at[val_pos, name] = ['unseen-heldout-only']
    d._remap_ids()
    specs, inputs = fixed_rec_list_inputs(d)
    cols = build_feature_columns(training_rows(d.interaction_df), d.model_cfg, d.col_types)
    din, _ = df_to_input(d.interaction_df, cols, d.model_cfg)
    np.testing.assert_array_equal(inputs[name]['ids'].numpy(), din[name])
    np.testing.assert_array_equal(inputs[name]['lengths'].numpy(), din[name+'_length'])
    assert int(inputs[name]['ids'][val_pos, 0]) == 0
    assert inputs[name]['mask'][val_pos, 0] and inputs[name]['lengths'][val_pos] == 1
    changed = copy.copy(d)
    changed.interaction_df = d.interaction_df.copy()
    changed.interaction_df.loc[changed.interaction_df['__split__'].eq('val'), 'rating'] ^= 1
    specs2, inputs2 = fixed_rec_list_inputs(changed)
    assert specs2 == specs
    for key in inputs[name]:
        torch.testing.assert_close(inputs[name][key], inputs2[name][key])
    model = RecSurrogate(len(d.category_maps_['user_id'])+1, len(d.category_maps_['item_id'])+1,
                         feat_dim=0, emb_dim=4, hidden=8, list_feature_specs=specs)
    out = model(torch.tensor(d.interaction_df.user_id.to_numpy()),
                torch.tensor(d.interaction_df.item_id.to_numpy()), None, fixed_lists=inputs)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.list_embeddings.parameters())


@pytest.mark.parametrize('device', DEVICES)
def test_explicit_lengths_preserve_oov_and_exclude_padding(device):
    emb = torch.nn.Embedding(3, 2, device=device)
    with torch.no_grad():
        emb.weight[0].fill_(2.)
    ids = torch.zeros(2, 3, device=device, dtype=torch.long)
    lengths = torch.tensor([1, 0], device=device)
    mask = torch.arange(3, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    pooled = RecSurrogate._pool(emb, ids, lengths, mask)
    torch.testing.assert_close(pooled, torch.tensor([[2., 2.], [0., 0.]], device=device))
    pooled.sum().backward()
    torch.testing.assert_close(emb.weight.grad[0], torch.ones(2, device=device))


def test_recent_queries_keep_full_training_context_and_match_deployment(tmp_path):
    d, cfg, _ = dataset(tmp_path)
    ctx, _, audit = prepare_rec_search(d, cfg, 42, 3)
    assert len(training_rows(d.interaction_df)) == 3
    assert len(d._rec_history_reference) == 18
    assert audit['history_reference_rows'] == 18
    assert d._rec_history_reference['__split__'].eq('train').all()
    op = d._load_op('CreateSequence')(**d._diffprep_sequence_step.params)
    # Independent deployment reconstruction on the complete legal training pool.
    reference = d._rec_history_reference.copy()
    full = op.transform(pd.concat([reference, d.interaction_df.loc[d.interaction_df['__split__'].eq('val')]
                                  .drop(columns='item_id_seq')], ignore_index=True))
    expected = full.set_index(['user_id', 'item_id', 'timestamp']).item_id_seq
    for row in d.interaction_df.itertuples():
        assert row.item_id_seq == expected.loc[(row.user_id, row.item_id, row.timestamp)]
        allowed = reference.loc[(reference.user_id == row.user_id) & (reference.timestamp < row.timestamp)]
        assert row.item_id_seq == allowed.sort_values('timestamp').item_id.tolist()[-20:]
    # Early validation queries retain legal history despite recent query sampling.
    val = d.interaction_df.loc[d.interaction_df['__split__'].eq('val')]
    early = val.loc[val.timestamp < training_rows(d.interaction_df).timestamp.min()]
    assert early.item_id_seq.map(len).gt(0).any()
    with pytest.raises(ValueError, match='training events only'):
        op.transform_with_history(d.interaction_df, reference.assign(__split__='val'))


def test_hpo_trials_zero_test_calls_and_final_one_with_fresh_checkpoint(tmp_path, monkeypatch):
    """Two mocked trials exercise orchestration, not a real HPO experiment."""
    from types import SimpleNamespace
    from scripts.final_model_hpo import tune_final_model
    from baselines.common.pipeline import Pipeline
    records = []
    class Evaluator:
        def __init__(self, **kwargs):
            self.enabled = True
            self._executor = SimpleNamespace(cleanup=lambda: None,
                get_model_config=lambda: {'train': {'metrics': ['auc']}})
        def set_evaluate_std_test(self, enabled):
            self.enabled = enabled
        def evaluate(self, pipeline, **kwargs):
            records.append(int(self.enabled))
            return SimpleNamespace(success=True, error=None, duration_seconds=0.,
                metrics={'auc': .5, 'rec_protocol': {'feature_schema_sha256': 'schema',
                                                   'category_maps_sha256': 'maps'}})
    monkeypatch.setattr('scripts.final_model_hpo.candidate_configs', lambda _: [{}, {'model_params': {'lr': .1}}])
    evaluator, summary = tune_final_model(Pipeline([]), output_dir=str(tmp_path),
                                          evaluator_class=Evaluator, model_name='DIN')
    assert records == [0, 0]
    assert evaluator._executor.expected_rec_schema_sha256 == 'schema'
    assert evaluator._executor.expected_category_maps_sha256 == 'maps'
    evaluator.evaluate(Pipeline([]))
    assert records == [0, 0, 1]


def test_side_snapshot_values_are_joined_without_asof_filter():
    """Documents the unresolved side-table gate; does NOT certify availability."""
    from dppbench.operators.integration.join_table import JoinTable
    frame = pd.DataFrame({'user_id': ['u'], 'item_id': ['i'], 'timestamp': [1], 'rating': [0]})
    users = pd.DataFrame({'user_id': ['u'], 'user_average_stars': [4.8], 'user_review_count': [100]})
    items = pd.DataFrame({'item_id': ['i'], 'item_stars': [4.9]})
    joined = JoinTable(method='rec', user_df=users, item_df=items).transform(frame)
    assert joined.user_average_stars.iloc[0] == 4.8
    assert joined.item_stars.iloc[0] == 4.9
    # Changing only metadata changes prediction input even with a fixed interaction pool.
    future = items.assign(item_stars=1.)
    assert JoinTable(method='rec', item_df=future).transform(frame).item_stars.iloc[0] == 1.


def test_kuairec_loader_selects_latest_snapshot_and_discards_time(monkeypatch):
    """Source-path reproduction of an unresolved availability gate, not real data."""
    from dppbench.tasks.kuairec.kuairec_data import KuairecData
    d = KuairecData(data_dir='unused-readonly-fixture')
    monkeypatch.setattr(d, '_file_path', lambda name: name)
    categories = pd.DataFrame({'video_id': [1], 'feat': ['[1, 2]']})
    daily = pd.DataFrame({c: [1, 1] for c in d.ITEM_DAILY_COLS})
    daily['date'] = ['2020-01-01', '2021-01-01']
    daily['visible_status'] = [0, 1]
    monkeypatch.setattr(pd, 'read_csv', lambda path, **kwargs:
        categories.copy() if str(path).endswith('item_categories.csv') else daily.copy())
    result = d._load_items()
    assert result.item_visible_status.iloc[0] == 1
    assert 'date' not in result


@pytest.mark.parametrize('task,target,feature', [
    ('movielens', 'rating', 'item_genres'),
    ('kuairec', 'watch_ratio', 'item_categories'),
    ('yelp', 'stars', 'item_categories')])
def test_two_epoch_recommendation_search_keeps_deployed_lists(tmp_path, monkeypatch, task, target, feature):
    from baselines.DiffPrep.diffprep import DiffPrep
    from baselines.common.data_loader import DATASET_REGISTRY, import_class
    d, cfg, std = dataset(tmp_path)
    d.col_types[feature] = 'categorical_list'
    d.item_df[feature] = [['a'], ['b'], ['a', 'b']]
    if target != 'rating':
        d.col_types[target] = d.col_types.pop('rating')
        d.interaction_df = d.interaction_df.rename(columns={'rating': target})
        for name in ('interaction_frozen.parquet', 'std_test.parquet'):
            pd.read_parquet(std/name).rename(columns={'rating': target}).to_parquet(std/name, index=False)
    if task == 'kuairec':
        d.col_types['user_friend_list'] = 'categorical_list'
        d.user_df = pd.DataFrame({'user_id': ['u1', 'u2'], 'user_friend_list': [['u2'], ['u1']]})
    def load(data):
        data.interaction_df = d.interaction_df.copy(deep=True)
        data.item_df = d.item_df.copy(deep=True)
        data.user_df = d.user_df.copy(deep=True) if d.user_df is not None else None
        data.col_types = dict(d.col_types)
        return data.interaction_df, data.item_df
    cls = import_class(DATASET_REGISTRY[task][1])
    monkeypatch.setattr(cls, 'load_data', load)
    root = Path(__file__).resolve().parents[1]
    runner = DiffPrep(str(root/'dppbench/tasks'/task), task, data_dir=d.data_dir,
        output_dir=str(tmp_path/'run'), n_epochs=2, small_n=3, batch_size=2,
        eval_full=False, verbose=False, device='cpu')
    result = runner.run()
    assert len(result['search_history']) == 2
    assert result['search_contract']['version'] == 5
    assert feature in result['search_contract']['fixed_list_features']
    if task == 'kuairec':
        assert 'user_friend_list' in result['search_contract']['fixed_list_features']
    assert result['search_contract']['train_rows'] == 3
    assert result['search_contract']['history_reference_rows'] > 3
    assert 'JoinTable' in result['final_pipeline_ops']
