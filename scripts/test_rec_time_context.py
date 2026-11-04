"""Synthetic-only clock compatibility; no downloads, search or full HPO."""
import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from dppbench.dataset import RecData
from dppbench.operators.integration.parse_date import ParseDate
from dppbench.operators.integration.sort_rows import SortRows
from dppbench.operators.feature_engineering.create_sequence import CreateSequence
from dppbench.rec_time_context import EVENT_CONTEXT, event_context


def data_and_frame(unit='s'):
    # Distinct instants within a day plus an equal-time training event.
    values = [1500000000, 1500000100, 1500000200, 1500000200, 1500000300, 1500000400]
    if unit in ('ms', 'ns'):
        values = np.asarray(values, dtype='int64') * (1000 if unit == 'ms' else 10**9)
    elif unit == 'utc':
        values = list(pd.to_datetime(values, unit='s', utc=True))
    d = RecData('synthetic_clock')
    d.model_cfg = {'feature': {'target_col': 'rating', 'time_col': 'timestamp',
        'seq_cols': {'item_id_seq': 'item_id'}, 'seq_max_len': 20},
        'model_params': {'embedding_dim': 4}, 'train': {'seed': 42}}
    if unit != 'utc':
        d.model_cfg['feature']['time_rules'] = {'timestamp': {'unit': unit, 'origin': 'unix'}}
    d.col_types = {'user_id': 'categorical', 'item_id': 'categorical',
        'timestamp': 'timestamp', 'rating': 'numeric', 'x': 'numeric'}
    f = pd.DataFrame({'user_id': ['u']*6, 'item_id': ['past', 'middle', 'tie', 'val', 'test', 'future'],
        'timestamp': values, 'rating': [0, 1, 0, 1, 0, 1], 'x': np.arange(6, dtype=float),
        '__split__': ['train', 'train', 'train', 'val', 'std_test', 'train']})
    d._rec_history_reference = f.loc[f.__split__.eq('train'), ['user_id','item_id','timestamp','__split__']].copy()
    return d, f


def transform(d, f, op):
    return d._transform_rec_dag_df(f, op, op.name)


@pytest.mark.parametrize('unit', ['s', 'ms', 'ns', 'utc'])
@pytest.mark.parametrize('format', [None, '%Y-%m-%d'])
def test_parse_date_uses_original_instant_not_candidate_dtype_or_day(unit, format):
    d, f = data_and_frame(unit)
    expected = CreateSequence().transform_with_history(f, d._rec_history_reference)
    history = d._rec_history_reference.copy(deep=True)
    parsed = transform(d, f, ParseDate(['timestamp'], target_format=format))
    result = transform(d, parsed, CreateSequence())
    assert dict(zip(result.item_id,result.item_id_seq)) == dict(zip(expected.item_id,expected.item_id_seq))
    assert result.loc[result.item_id.eq('val'), 'item_id_seq'].iloc[0] == ['past', 'middle']
    assert result.loc[result.item_id.eq('test'), 'item_id_seq'].iloc[0] == ['past', 'middle', 'tie']
    pd.testing.assert_frame_equal(d._rec_history_reference, history)
    assert EVENT_CONTEXT not in d.col_types


def test_sort_filter_replicate_copy_and_pickle_keep_row_aligned_context():
    from dppbench.operators.preprocessing.oversample import Oversample
    d, f = data_and_frame()
    baseline = dict(zip(f.item_id, CreateSequence().transform_with_history(f,d._rec_history_reference).item_id_seq))
    parsed = transform(d, f, ParseDate(['timestamp'], target_format='%Y-%m-%d'))
    for frame in (parsed.copy(deep=False), copy.deepcopy(parsed), pickle.loads(pickle.dumps(parsed))):
        frame = transform(d, frame, SortRows(by=['item_id']))
        frame = frame.loc[~frame.item_id.eq('past')].reset_index(drop=True)
        frame = transform(d, frame, Oversample(target_col='rating', method='random'))
        result = transform(d, frame, CreateSequence())
        assert all(seq == baseline[item] for item, seq in zip(result.item_id, result.item_id_seq))
        # Filtered past query remains part of the immutable history reference.
        assert result.loc[result.item_id.eq('val'), 'item_id_seq'].iloc[0] == ['past', 'middle']


def test_repeated_date_parse_does_not_recapture_coarse_clock():
    d, f = data_and_frame()
    a = transform(d, f, ParseDate(['timestamp'], target_format='%Y-%m-%d'))
    b = transform(d, a, ParseDate(['timestamp']))
    assert event_context(a) == event_context(b)
    result = transform(d, b, CreateSequence())
    assert result.loc[result.item_id.eq('val'),'item_id_seq'].iloc[0] == ['past', 'middle']


def test_default_deduplicate_and_label_encoding_do_not_consume_context():
    from dppbench.operators.cleaning.deduplicate import Deduplicate
    from dppbench.operators.preprocessing.label_encode import LabelEncode
    d, f = data_and_frame()
    f = f.iloc[:2].copy(); f['item_id']='same'; f['rating']=0; f['x']=1.
    parsed = transform(d, f, ParseDate(['timestamp'], target_format='%Y-%m-%d'))
    assert parsed.timestamp.nunique() == 1
    reduced = transform(d, parsed, Deduplicate())
    assert len(reduced) == 1  # Context must not keep otherwise duplicate rows.
    before = event_context(reduced)
    encoded = transform(d, reduced, LabelEncode())
    assert event_context(encoded) == before


def test_metadata_is_not_a_baseline_feature_or_din_input_even_if_misregistered():
    from baselines.common.data_ctx import infer_data_context
    from dppbench.models.inputs import build_feature_columns
    d, f = data_and_frame()
    d.interaction_df = transform(d, f, ParseDate(['timestamp']))
    context = infer_data_context('synthetic_clock', d, task_type='rec')
    assert EVENT_CONTEXT not in context.tables['interaction'].columns
    d.interaction_df = transform(d, d.interaction_df, CreateSequence())
    d._remap_ids()
    d.col_types[EVENT_CONTEXT] = 'numeric'  # Existing prefix guard must still win.
    columns = build_feature_columns(d.interaction_df, d.model_cfg, d.col_types)
    assert EVENT_CONTEXT not in [column.name for column in columns]


def test_validation_test_label_flip_does_not_change_history_features():
    d, f = data_and_frame()
    changed = f.copy(); mask=~changed.__split__.eq('train'); changed.loc[mask,'rating']=1-changed.loc[mask,'rating']
    def encode(frame):
        return transform(d,transform(d,frame,ParseDate(['timestamp'])),CreateSequence()).item_id_seq.tolist()
    assert encode(f) == encode(changed)


def test_corrupt_or_explicitly_selected_metadata_fails_closed():
    from dppbench.operators.feature_engineering.frequency_encode import FrequencyEncode
    d, f = data_and_frame()
    parsed = transform(d, f, ParseDate(['timestamp']))
    with pytest.raises(ValueError, match='cannot select'):
        transform(d, parsed, FrequencyEncode([EVENT_CONTEXT]))
    parsed[EVENT_CONTEXT]=0
    with pytest.raises(ValueError, match='event context'):
        transform(d, parsed, CreateSequence())


def test_non_parse_pipeline_has_no_added_metadata():
    d, f = data_and_frame()
    result = transform(d, f, CreateSequence())
    assert EVENT_CONTEXT not in result


def test_timestamp_drop_and_cast_preserve_original_clock():
    from dppbench.operators.integration.drop_columns import DropColumns
    from dppbench.operators.integration.cast_type import CastType
    for op in (DropColumns(['timestamp']), CastType({'timestamp':'str'})):
        d, f = data_and_frame()
        result = transform(d, transform(d,f,op), CreateSequence())
        assert result.loc[result.item_id.eq('val'),'item_id_seq'].iloc[0] == ['past','middle']


def test_strict_statistics_use_same_original_clock_after_day_formatting():
    from dppbench.operators.feature_engineering.aggregate_group_feature import AggregateGroupFeature
    from dppbench.operators.feature_engineering.create_lag_feature import CreateLagFeature
    from dppbench.operators.feature_engineering.create_rolling_feature import CreateRollingFeature
    d, f = data_and_frame()
    parsed = transform(d, f, ParseDate(['timestamp'], target_format='%Y-%m-%d'))
    for make in (lambda: AggregateGroupFeature(['user_id'],['x'],['mean']),
                 lambda: CreateLagFeature('x',[1],group_cols=['user_id']),
                 lambda: CreateRollingFeature('x',[2],['mean'],group_cols=['user_id'])):
        a=transform(d,f,make());b=transform(d,parsed,make())
        outputs=[c for c in a if c not in f]
        pd.testing.assert_frame_equal(a.set_index('item_id')[outputs].sort_index(),
                                      b.set_index('item_id')[outputs].sort_index())


@pytest.mark.parametrize('name', ['AmazonBeauty','Movielens','Yelp','KuaiRec','kuairec'])
def test_known_loader_clock_has_explicit_unix_seconds(name):
    d,f=data_and_frame();d.name=name;d.model_cfg['feature'].pop('time_rules')
    result=transform(d,f,ParseDate(['timestamp']))
    assert result.timestamp.dt.year.eq(2017).all()
    assert transform(d,result,CreateSequence()).loc[lambda x:x.item_id.eq('val'),'item_id_seq'].iloc[0]==['past','middle']


def test_unknown_numeric_clock_is_not_guessed():
    d,f=data_and_frame();d.model_cfg['feature'].pop('time_rules')
    with pytest.raises(ValueError,match='declared'):
        transform(d,f,ParseDate(['timestamp']))


def test_real_relation_dag_drops_internal_context_before_model_mapping():
    d,f=data_and_frame();d.interaction_df=f
    dag={'dag':{'sources':[{'id':'s0','table':'interaction'}],
        'ops':[{'id':'o1','op':'ParseDate','prev':['s0'],'params':{'cols':['timestamp'],'target_format':'%Y-%m-%d'}},
               {'id':'o2','op':'SortRows','prev':['o1'],'params':{'by':['item_id']}},
               {'id':'o3','op':'CreateSequence','prev':['o2'],'params':{}}],
        'train':{'prev':['o3']}}}
    d._run_pre_process_relation_dag(dag)
    assert 'item_id_seq' in d.interaction_df and EVENT_CONTEXT not in d.interaction_df
    assert EVENT_CONTEXT not in d.category_maps_


@pytest.mark.parametrize('encoder', ['label', 'onehot'])
def test_fitted_encoder_can_replay_without_treating_context_as_selector(encoder):
    from dppbench.operators.preprocessing.label_encode import LabelEncode
    from dppbench.operators.preprocessing.one_hot_encode import OneHotEncode
    d,f=data_and_frame();parsed=transform(d,f,ParseDate(['timestamp']))
    op=LabelEncode() if encoder=='label' else OneHotEncode(['item_id'])
    first=transform(d,parsed,op)
    second=transform(d,parsed,op)
    assert event_context(first)==event_context(second)
