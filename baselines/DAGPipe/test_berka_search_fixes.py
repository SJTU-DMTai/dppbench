"""Relational search contracts behind the Berka seed-43 discrepancies."""
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from baselines.AlphaClean.parameter_sampler import ParameterSampler
from baselines.AlphaClean.repair import Repair, repairs_to_pipeline
from baselines.AlphaClean.searcher import AlphaCleanSearcher, ScoredPipeline
from baselines.common.data_ctx import infer_data_context
from baselines.DiffPrep.data_protocol import prepare_tabular_search
from baselines.DiffPrep.discretizer import discretize
from baselines.DiffPrep.search_space import ContinuousPipeline
from baselines.DAGPipe.runtime import limited_layers, make_model, reference_forward
from dppbench.dataset import TabularData
from scripts.diagnose_dagpipe_berka import representation_diagnostics


def fixture_data():
    data = TabularData("relational_fixture")
    data.id_col, data.target_col = "loan_id", "target"
    data.model_cfg = {"model_params": {"task": "binary"}, "train": {"val_ratio": .2}}
    data.train_df = pd.DataFrame({"loan_id": range(40), "account_id": range(40),
                                  "amount": np.arange(40.) + 100, "target": [0, 1] * 20})
    data.auxiliary_dfs = {
        "account": pd.DataFrame({"account_id": range(40), "district_id": np.arange(40) % 3}),
        "trans": pd.DataFrame({"account_id": np.repeat(np.arange(40), 2),
                               "balance": np.arange(80.)}),
        "unjoinable": pd.DataFrame({"other_id": [1, 2], "foo": [3., 4.]}),
    }
    return data


def test_alphaclean_join_seeds_cover_only_valid_relations():
    ctx = infer_data_context("fixture", fixture_data(), "tabular")
    first = ParameterSampler(ctx, seed=43).join_seeds()
    second = ParameterSampler(ctx, seed=43).join_seeds()
    assert [r.params for r in first] == [r.params for r in second]
    assert {r.params['aux_df']: r.params['method'] for r in first} == {
        '$account': 'key', '$trans': 'agg'}
    sampler = ParameterSampler(ctx, seed=43)
    for _ in range(40):
        p = sampler.sample_for_op('JoinTable').params
        assert p['method'] == ('key' if p['aux_df'] == '$account' else 'agg')
        assert p['key_col'] == 'account_id'


def test_alphaclean_replaces_join_even_at_length_limit():
    ctx = infer_data_context("fixture", fixture_data(), "tabular")
    sampler = ParameterSampler(ctx, seed=43)
    joins = {r.params['aux_df']: r for r in sampler.join_seeds()}
    initial = repairs_to_pipeline([joins['$account']], ctx, rng=random.Random(43))
    sampler.sample = lambda n: [joins['$trans']]
    class Evaluator:
        def evaluate(self, pipe):
            source = next(s.params['aux_df'] for s in pipe.steps if s.op == 'JoinTable')
            return SimpleNamespace(fitness=1. if source == '$trans' else .1, metrics={})
    pruner = SimpleNamespace(predict=lambda r: True, add_pipeline_examples=lambda *a: None,
                             fit_if_ready=lambda: None, stats={})
    search = AlphaCleanSearcher(Evaluator(), ctx, sampler, pruner, n_iters=1,
                               max_pipeline_len=len(initial.steps), verbose=False)
    def seed():
        search.frontier = [ScoredPipeline(initial, .1)]
    search._seed_frontier = seed
    best = search.run()[0].pipeline
    assert len(best.steps) == len(initial.steps)
    assert next(s.params['aux_df'] for s in best.steps if s.op == 'JoinTable') == '$trans'


def test_diffprep_joins_search_inputs_and_replays_identical_join(tmp_path):
    data = fixture_data()
    frozen = tmp_path / 'train_frozen.parquet'
    data.train_df.to_parquet(frozen, index=False)
    data.data_dir = str(tmp_path)
    data._std_test_dir = lambda _: str(tmp_path)
    ctx = infer_data_context('fixture', data, 'tabular')
    splits, joined_ctx, steps = prepare_tabular_search(data, data.model_cfg, ctx, 43)
    assert len(steps) == 1
    assert set(splits['train'].loan_id).isdisjoint(splits['val'].loan_id)
    added = set(splits['train']) - {'loan_id', 'account_id', 'amount', 'target'}
    assert added
    cont = ContinuousPipeline([], n_features=1, feature_names=['amount'])
    exported = discretize(cont, [], joined_ctx, rng=random.Random(999), structural_steps=steps)
    assert [s.params for s in exported.steps if s.op == 'JoinTable'] == [steps[0].params]
    # A different export RNG cannot switch to a relation never seen in search.
    assert added.issubset(set(joined_ctx.numeric_cols) | set(joined_ctx.categorical_cols))


def test_berka_dag_without_polynomial_preserves_feature_order():
    # At max_len=10, two joins + input encoding leave seven affine nodes.
    torch.manual_seed(43)
    layers = limited_layers(7)
    assert all('PolynomialTransform' not in layer for layer in layers)
    model = make_model(layers, 3)
    x = torch.randn(40, 3, dtype=torch.float64)
    with torch.no_grad():
        y = reference_forward(model, x, x[:16])
    report = representation_diagnostics(pd.DataFrame(x.numpy()), pd.DataFrame(y.numpy()))
    assert all(r['order_reversals'] == 0 for r in report.values())
