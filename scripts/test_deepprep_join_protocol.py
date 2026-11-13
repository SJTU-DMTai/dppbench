"""Regressions for the downloaded Amazon Beauty DeepPrep transcript."""
import random
from unittest.mock import Mock

import pytest

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, PipelineStep, assign_dag_structure, check_pipeline
from baselines.DeepPrep.tree_agent import TreeAgent, ChainParseError, chain_to_steps, parse_response
from baselines.DeepPrep.tree_node import canonical_path


def context():
    return DataContext('rec', 'amazon_beauty', 'interaction', secondary_tables=['item'],
        tables={
            'interaction': TableContext('interaction', 'primary',
                dtypes={'user_id': 'object', 'item_id': 'object', 'timestamp': 'int64', 'rating': 'int8'}),
            'item': TableContext('item', 'secondary', dtypes={'item_id': 'object', 'item_brand': 'string'}),
        })


def parse(chain):
    return chain_to_steps(chain, context(), random.Random(42), strict_instances=True)


def test_logged_mixed_join_modes_rejected_before_execution():
    with pytest.raises(ChainParseError, match='key/agg modes require aux_df'):
        parse("JoinTable(method='key', item_df='item', item_col='item_id', key_col='item_id', how='left', prefix='')")


@pytest.mark.parametrize('call', [
    "JoinTable(method='key', aux_df='item', key_col='item_id')",
    "JoinTable(method='rec', item_df='item', item_col='item_id')",
])
def test_explicit_plain_table_reference_connects_two_sources(call):
    step = parse(call)[0]
    assert '$item_df' in step.params.values()
    pipeline = Pipeline([step])
    assign_dag_structure(pipeline, context())
    assert len(pipeline.steps[0].prev) == 2
    assert check_pipeline(pipeline, context(), require_mandatory=False).valid


def test_join_hint_preserves_materialized_sequence_and_forms_legal_pipeline():
    ctx = context()
    agent = TreeAgent(Mock(), Mock(), ctx, verbose=False)
    agent.tree.add_root(b'root', 'root')
    seq = parse("CreateSequence(item_col='item_id', max_len=50, seq_col='item_id_seq', time_col='timestamp', user_col='user_id')")
    agent.tree.add_child(seq, b'sequence', 'sequence')
    hint = agent._revision_hint()
    continuation = next(line.removeprefix('Continuation syntax: ') for line in hint.splitlines()
                        if line.startswith('Continuation syntax: '))
    action = parse_response(continuation)
    parent, new_steps = agent._resolve_expansion(action.chain_str)
    assert parent.accumulated_steps == seq
    assert [s.op for s in new_steps] == ['JoinTable']
    pipeline = Pipeline(seq + new_steps)
    assign_dag_structure(pipeline, ctx)
    assert check_pipeline(pipeline, ctx).valid
    assert 'do not lowercase' in hint
    assert canonical_path(seq) in continuation


def test_logged_prose_is_not_silently_converted_to_an_operator():
    with pytest.raises(ChainParseError, match='keyword arguments'):
        parse('CreateSequence (to build user-item sequences)')
