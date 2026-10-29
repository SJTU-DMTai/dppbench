"""Small correctness fixtures: never download data or run benchmark experiments."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from dppbench.dataset import RecData
from dppbench.rec_protocol import training_rows, assert_disjoint
from dppbench.models.inputs import build_feature_columns, df_to_input, SparseFeat, DenseFeat
from dppbench.models.din import DIN
from baselines.common.data_ctx import infer_data_context
from baselines.common.pipeline import validate_step_params


CFG = {'feature': {'target_col': 'rating', 'seq_max_len': 3,
                  'seq_cols': {'item_id_seq': 'item_id'}},
       'train': {'val_ratio': .25, 'seed': 42},
       'model_params': {'embedding_dim': 4, 'task': 'binary'}}


def fixture():
    d = RecData('amazon_beauty')
    d.set_model_config(copy.deepcopy(CFG))
    d.col_types = {'user_id': 'categorical', 'item_id': 'categorical',
                   'brand': 'categorical', 'rating': 'categorical',
                   'timestamp': 'timestamp', 'x': 'numeric'}
    d.interaction_df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u1', 'u2', 'u1', 'u1', 'new'],
        'item_id': ['i1', 'i2', 'i1', 'i2', 'i2', 'i1', 'new'],
        'brand': ['a', 'b', 'a', 'b', 'b', 'b', 'unseen'],
        'rating': [1, 0, 1, 0, 1, 0, 1],
        'timestamp': [1, 3, 5, 1, 2, 6, 7],
        'x': [10., 30., 50., 7., 2000., 9000., 8000.],
        '__split__': ['train']*4 + ['val', 'std_test', 'std_test']})
    return d


def apply(d, name, **params):
    op = d._load_op(name)(**params)
    d.interaction_df = d._transform_rec_dag_df(d.interaction_df, op, name)
    return d.interaction_df


@pytest.mark.parametrize('op,params', [
    ('FrequencyEncode', {'cols': ['rating']}), ('HashEncode', {'cols': ['rating']}),
    ('OneHotEncode', {'cols': ['rating']}), ('TargetEncode', {'cols': ['rating']}),
    ('LabelEncode', {'cols': ['rating']}), ('CrossFeature', {'cols': ['brand', 'rating']}),
    ('CreateFeature', {'source_cols': ['rating'], 'output_col': 'copy'}),
    ('CreateLagFeature', {'target_col': 'rating', 'lags': [1]}),
    ('RenameColumn', {'rename_map': {'rating': 'hidden_label'}}),
])
def test_target_protected_in_validator_and_executor(op, params):
    d = fixture()
    ctx = infer_data_context('amazon_beauty', d, task_type='rec')
    assert validate_step_params(op, params, ctx)
    with pytest.raises(ValueError, match='supervised target|supervised targets'):
        apply(d, op, **params)


@pytest.mark.parametrize('name,params', [
    ('TargetEncode', {'cols': ['brand']}), ('FrequencyEncode', {'cols': ['brand']}),
    ('OneHotEncode', {'cols': ['brand']}), ('HashEncode', {'cols': ['brand']}),
    ('AggregateGroupFeature', {'group_cols': ['user_id'], 'agg_cols': ['x'], 'agg_funcs': ['mean', 'count']}),
    ('CreateLagFeature', {'target_col': 'x', 'lags': [1], 'group_cols': ['user_id'], 'time_col': 'timestamp'}),
    ('CreateRollingFeature', {'target_col': 'x', 'windows': [2], 'group_cols': ['user_id'], 'time_col': 'timestamp'}),
])
def test_heldout_label_flip_cannot_change_features(name, params):
    a, b = fixture(), fixture()
    mask = b.interaction_df['__split__'].ne('train')
    b.interaction_df.loc[mask, 'rating'] = 1-b.interaction_df.loc[mask, 'rating']
    fa, fb = apply(a, name, **params), apply(b, name, **params)
    pd.testing.assert_frame_equal(fa.drop(columns='rating'), fb.drop(columns='rating'))


def test_strict_training_history_no_heldout_updates_or_equal_time_peeking():
    d = fixture()
    f = apply(d, 'CreateLagFeature', target_col='x', lags=[1], group_cols=['user_id'], time_col='timestamp')
    assert f.loc[4, 'x_lag_1'] == 10  # val at 2 cannot use train at 3 or itself
    assert f.loc[5, 'x_lag_1'] == 50  # test ignores val's x=2000
    d = fixture()
    f = apply(d, 'AggregateGroupFeature', group_cols=['user_id'], agg_cols=['x'], agg_funcs=['count', 'mean'])
    assert f.loc[4, 'x_count'] == 1
    assert f.loc[5, 'x_mean'] == 30
    d = fixture()
    d.interaction_df.loc[1, 'timestamp'] = 1
    f = apply(d, 'CreateSequence', max_len=3)
    assert f.loc[0, 'item_id_seq'] == f.loc[1, 'item_id_seq'] == []
    assert f.loc[4, 'item_id_seq'] == ['i1', 'i2']


def test_shared_categories_oov_and_frozen_schema():
    d = fixture()
    apply(d, 'CreateSequence', max_len=3)
    d._remap_ids()
    f = d.interaction_df
    assert f.loc[1, 'brand'] == f.loc[4, 'brand'] == f.loc[5, 'brand']
    assert f.loc[6, 'brand'] == f.loc[6, 'user_id'] == f.loc[6, 'item_id'] == 0
    train = training_rows(f)
    hpo = build_feature_columns(train, d.model_cfg, d.col_types)
    final = build_feature_columns(training_rows(f), d.model_cfg, d.col_types)
    assert hpo == final
    assert all(c.vocabulary_size == len(d.category_maps_[c.name])+1 for c in hpo if isinstance(c, SparseFeat))
    for split in ('train', 'val', 'std_test'):
        x, _ = df_to_input(f.loc[f['__split__'].eq(split)], hpo, d.model_cfg)
        assert x['brand'].dtype == np.int64


def test_object_fallback_and_wrong_numeric_metadata_fail_loudly():
    frame = pd.DataFrame({'brand': ['a'], 'rating': [1]})
    with pytest.raises(ValueError, match='train-fitted'):
        df_to_input(frame, [SparseFeat('brand', 3, 4)], CFG)
    with pytest.raises(ValueError, match='nonnumeric'):
        df_to_input(frame, [DenseFeat('brand', 1)], CFG)


@pytest.mark.parametrize('name,params,output,kind', [
    ('HashEncode', {'cols': ['brand']}, 'h_brand', 'categorical'),
    ('CrossFeature', {'cols': ['brand', 'user_id']}, 'brand_user_id', 'categorical'),
    ('CreateFeature', {'source_cols': ['brand', 'user_id'], 'output_col': 'joined', 'method': 'concat'}, 'joined', 'categorical'),
    ('ExtractDateTimeFeature', {'cols': ['timestamp'], 'features': ['year']}, 'timestamp_year', 'numeric'),
    ('AggregateGroupFeature', {'group_cols': ['user_id'], 'agg_cols': ['x'], 'agg_funcs': ['count']}, 'x_count', 'numeric'),
    ('TransformPower', {'cols': ['x'], 'output_cols': ['log_x']}, 'log_x', 'numeric'),
    ('ExtractTextEmbedding', {'cols': ['brand'], 'dim': 2}, 'brand_emb_0', 'numeric'),
    ('CreateLagFeature', {'target_col': 'x', 'lags': [1], 'time_col': 'timestamp'}, 'x_lag_1', 'numeric'),
    ('CreateRollingFeature', {'target_col': 'x', 'windows': [2], 'time_col': 'timestamp'}, 'x_roll_mean_2', 'numeric'),
    ('CreatePolynomialFeature', {'cols': ['x']}, 'poly_xp2', 'numeric'),
    ('ReduceDimension', {'cols': ['x'], 'n_components': 1}, 'pca_0', 'numeric'),
    ('ExtractTextFeature', {'cols': ['user_id'], 'method': 'tfidf'}, 'user_id_tfidf_u1', 'numeric'),
    ('ExtractGraphFeature', {'source_col': 'user_id', 'target_col': 'item_id', 'features': ['degree']}, 'graph_user_id_degree', 'numeric'),
    ('SplitColumn', {'col': 'brand'}, 'brand_0', 'categorical'),
    ('RenameColumn', {'rename_map': {'x': 'renamed'}}, 'renamed', 'numeric'),
])
def test_new_features_reach_real_din_tensor_and_computation(name, params, output, kind):
    d = fixture()
    apply(d, name, **params)
    assert d.col_types[output] == kind
    apply(d, 'CreateSequence', max_len=3)
    d._remap_ids()
    f = d.interaction_df
    fc = build_feature_columns(training_rows(f), CFG, d.col_types)
    x, _ = df_to_input(f, fc, CFG)
    assert output in x
    model = DIN(fc, dnn_hidden_units=(8, 4), att_hidden_size=(4, 2), seed=42)
    model.eval()
    batch = torch.tensor(np.concatenate([x[c].reshape(len(f), -1) for c in model.feature_index], axis=1),
                         dtype=torch.float32, requires_grad=True)
    model(batch).sum().backward()
    assert output in model.feature_index
    if kind == 'categorical':
        assert model.embedding_dict[output].weight.grad is not None
    else:
        lo, hi = model.feature_index[output]
        assert batch.grad[:, lo:hi].abs().sum() > 0


def test_weights_and_diagnostic_flags_are_never_predictors():
    d = fixture()
    f = d.interaction_df
    f['sample_weight'], f['rating_corrected'] = .2, 1
    d.col_types.update(sample_weight='numeric', rating_corrected='numeric')
    names = [c.name for c in build_feature_columns(f, CFG, d.col_types)]
    assert 'sample_weight' not in names and 'rating_corrected' not in names


def test_identity_audit_does_not_need_labels():
    d = fixture()
    f = d.interaction_df
    audit = assert_disjoint(training_rows(f), f.loc[f['__split__'].eq('std_test')].drop(columns='rating'))
    assert audit['train_identity_sha256']
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(training_rows(f), training_rows(f).drop(columns='rating'))


def test_generated_feature_cannot_overwrite_target_or_promote_diagnostics():
    d = fixture()
    d.interaction_df['rat'] = ['a']*len(d.interaction_df)
    with pytest.raises(ValueError, match='supervised target'):
        apply(d, 'FrequencyEncode', cols=['rat'], suffix='ing')
    with pytest.raises(ValueError, match='supervised target'):
        apply(d, 'CreateFeature', source_cols=['x'], output_col='sample_weight')


def test_correct_label_never_changes_heldout_ground_truth():
    d = fixture()
    d.interaction_df['probability'] = .99
    before = d.interaction_df.loc[d.interaction_df['__split__'].ne('train'), 'rating'].tolist()
    f = apply(d, 'CorrectLabel', label_col='rating', pred_probs='probability', strategy='flip')
    assert f.loc[f['__split__'].ne('train'), 'rating'].tolist() == before


def test_training_weight_changes_loss_not_model_features():
    from dppbench.models.base_model import BaseModel
    class Toy(BaseModel):
        def __init__(self):
            super().__init__(seed=42)
            self.w = torch.nn.Parameter(torch.zeros(()))
        def forward(self, x):
            return torch.sigmoid(self.w*x[:, 0])
    def trained(weights):
        model = Toy()
        model.compile(optimizer='sgd', lr=.1)
        model.fit([np.ones((2, 1), dtype=np.float32)], np.array([0., 1.], dtype=np.float32),
                  epochs=1, batch_size=2, shuffle=False, verbose=0, sample_weight=weights)
        return model.w.item()
    assert trained([1., 5.]) > 0
    assert trained([5., 1.]) < 0
    assert trained([1., 1.]) == 0


def test_amazon_exact_metadata_duplicates_are_normalized_conflicts_rejected():
    from dppbench.tasks.amazon_beauty.amazon_beauty_data import AmazonBeautyData
    d = AmazonBeautyData('unused')
    d.set_model_config({'feature': {'target_col': 'rating', 'label_rule': {'mode': 'ge', 'threshold': 3}}})
    d._download_file = lambda _, name: name
    interactions = pd.DataFrame({'reviewerID': ['u1'], 'asin': ['i1'], 'unixReviewTime': [1], 'overall': [5]})
    metadata = pd.DataFrame({'asin': ['i1', 'i1'], 'brand': ['a', 'a']})
    d._load_gz_json = lambda name: metadata if name.startswith('meta_') else interactions
    d.load_data()
    assert d.item_df.item_id.is_unique and len(d.item_df) == 1
    metadata.loc[1, 'brand'] = 'conflict'
    with pytest.raises(ValueError, match='conflicting metadata'):
        d.load_data()


def test_final_data_manifest_detects_artifact_content_changes(tmp_path):
    from dppbench.rec_protocol import frozen_file_manifest
    path = tmp_path/'interaction_frozen.parquet'
    path.write_bytes(b'training-artifact-v1')
    first = frozen_file_manifest(tmp_path)
    path.write_bytes(b'training-artifact-v2')
    second = frozen_file_manifest(tmp_path)
    assert first[path.name]['sha256'] != second[path.name]['sha256']
