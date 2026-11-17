"""Structural adapter tests: no production dataset or model training required."""
import copy
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dppbench.dataset import TabularData
from baselines.common.data_ctx import infer_data_context
from baselines.common.pipeline import make_step
from baselines.DiffPrep.data_protocol import select_join, prepare_tabular_search, assert_join_replay
from baselines.DiffPrep.discretizer import discretize
from baselines.DiffPrep.search_space import ContinuousPipeline


def fixture(tmp_path):
    data = TabularData('fixture')
    data.id_col, data.target_col = 'loan_id', 'target'
    data.data_dir = str(tmp_path)
    data._std_test_dir = lambda _: str(tmp_path)
    data.model_cfg = {'train': {'val_ratio': .2, 'seed': 43}, 'model_params': {'task': 'binary'}}
    data.train_df = pd.DataFrame({'loan_id': range(20), 'account_id': range(20),
                                  'amount': range(20), 'target': [0, 1] * 10})
    data.train_df.to_parquet(tmp_path / 'train_frozen.parquet', index=False)
    data.auxiliary_dfs = {
        'account': pd.DataFrame({'account_id': range(20), 'district_id': range(20)}),
        'order': pd.DataFrame({'account_id': list(range(20)) * 2, 'amount': [2.] * 20 + [6.] * 20}),
    }
    return data, infer_data_context('fixture', data, 'tabular')


def test_default_preserves_seeded_choice_and_global_rng(tmp_path):
    _, ctx = fixture(tmp_path)
    state = random.getstate()
    for seed in (42, 43, 44):
        assert select_join(ctx, seed).params == make_step('JoinTable', ctx, random.Random(seed)).params
    assert random.getstate() == state


def test_explicit_join_and_yaml_replay_independent_of_export_rng(tmp_path):
    data, ctx = fixture(tmp_path)
    config = {'table': 'order', 'key_col': 'account_id', 'method': 'agg', 'prefix': 'ORD', 'max_cols': 2}
    original = copy.deepcopy(config)
    splits, ctx, steps = prepare_tabular_search(data, data.model_cfg, ctx, 43, join_config=config)
    assert config == original
    assert splits['train']['ORD_amount_mean'].eq(4).all()
    assert splits['val']['ORD_amount_mean'].eq(4).all()
    cont = ContinuousPipeline([], n_features=1, feature_names=['amount'])
    for seed in (0, 999):
        pipe = discretize(cont, [], ctx, random.Random(seed), structural_steps=steps)
        assert_join_replay(steps, pipe)
        assert pipe.steps[0].params['aux_df'] == '$order'
    # Exercise actual serialized DAG source resolution and joins.
    data.train_df = pd.read_parquet(tmp_path / 'train_frozen.parquet')
    data._skip_std_test_loading = True
    path = tmp_path / 'pipeline.yaml'
    path.write_text(pipe.to_yaml(), encoding='utf-8')
    data.run_pre_process(str(path))
    actual = data._prepared_splits()
    for name in ('train', 'val'):
        pd.testing.assert_frame_equal(actual[name][list(splits[name])], splits[name])
    for source in pipe.sources:
        if source['table'] == 'order':
            source['table'] = 'account'
    with pytest.raises(ValueError, match='source differs'):
        assert_join_replay(steps, pipe)


@pytest.mark.parametrize('config', [
    {'table': 'missing', 'key_col': 'account_id', 'method': 'agg'},
    {'table': 'order', 'key_col': 'account_id', 'method': 'key'},
    {'table': 'order', 'key_col': 'target', 'method': 'agg'},
    {'table': 'order', 'key_col': 'account_id', 'method': 'agg', 'max_cols': 0},
    {'table': 'order', 'key_col': 'account_id', 'method': 'agg', 'how': 'inner'},
])
def test_bad_explicit_configuration_fails_instead_of_random_fallback(tmp_path, config):
    _, ctx = fixture(tmp_path)
    with pytest.raises(ValueError):
        select_join(ctx, 43, config)


def test_config_file_scopes_override_to_dataset_and_old_config_still_loads(tmp_path):
    from baselines.DiffPrep.diffprep import DiffPrep
    source = Path(__file__).with_name('config.yaml')
    cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
    cfg['join_overrides'] = {'berka': {'table': 'order', 'key_col': 'account_id', 'method': 'agg'}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    model = DiffPrep(task_dir='unused', data_name='berka', config_path=str(path), output_dir=str(tmp_path / 'out'))
    assert model.join_overrides['berka']['table'] == 'order'
    assert model.join_overrides.get('home_credit') is None
    del cfg['join_overrides']
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    model = DiffPrep(task_dir='unused', data_name='berka', config_path=str(path), output_dir=str(tmp_path / 'out'))
    assert model.join_overrides == {}


def test_full_search_receives_explicit_adapter_and_records_replay(tmp_path, monkeypatch):
    import json
    from baselines.DiffPrep import diffprep
    from baselines.DiffPrep.test_search_protocol import test_berka_search_reproducible_without_reading_std_test
    original = diffprep.DiffPrep
    def explicit_model(*args, **kwargs):
        kwargs['join_overrides'] = {'berka': {
            'table': 'account', 'key_col': 'account_id', 'method': 'key'}}
        return original(*args, **kwargs)
    monkeypatch.setattr(diffprep, 'DiffPrep', explicit_model)
    test_berka_search_reproducible_without_reading_std_test(tmp_path, monkeypatch)
    contract = json.loads((tmp_path / 'first/search_contract.json').read_text(encoding='utf-8'))
    assert contract['join_replay_checked'] is True
    assert contract['join_selection']['mode'] == 'explicit'
    assert contract['search_joins'] == [{'method': 'key', 'key_col': 'account_id',
                                       'aux_df': '$account', 'how': 'left'}]
