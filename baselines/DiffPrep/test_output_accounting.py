"""Accounting must preserve legacy training results and random streams."""
import random
import numpy as np
import pytest
import torch
from baselines.common.search_cost import SearchCost, search_cost_scope

@pytest.mark.parametrize('second_order', [False, True])
def test_diffprep_real_gradients_counts_and_rng_parity(second_order):
    from baselines.DiffPrep.trainer import DiffPrepTrainer

    class Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tau = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
            self.theta = None

    def run(observe):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        cont = Architecture()
        model = torch.nn.Linear(1, 1, dtype=torch.float64)

        def loss(c, m):
            x = torch.rand(8, 1, dtype=torch.float64)
            return ((m(x) - c.tau * x - random.random() - np.random.rand()) ** 2).mean()

        trainer = DiffPrepTrainer(cont, model, loss, lambda c, m: (loss(c, m), 0.),
                                  lambda c: [], n_epochs=2, second_order=second_order, verbose=False)
        cost = SearchCost('DiffPrep')
        if observe:
            with search_cost_scope(cost):
                trainer.train()
        else:
            trainer.train()
        state = [p.detach().clone() for p in list(cont.parameters()) + list(model.parameters())]
        rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
        return state, rng, cost

    plain, plain_rng, _ = run(False)
    observed, observed_rng, cost = run(True)
    assert all(torch.equal(a, b) for a, b in zip(plain, observed))
    assert torch.equal(plain_rng[0], observed_rng[0])
    assert plain_rng[1] == observed_rng[1]
    assert np.array_equal(plain_rng[2][1], observed_rng[2][1])
    fields = cost.fields()
    assert fields['search_proxy_train_loss_calls'] == (8 if second_order else 4)
    assert fields['search_proxy_val_loss_calls'] == 2
    assert fields['search_gradient_calls'] == (12 if second_order else 6)
    assert fields['search_optimizer_steps'] == 4
    assert fields['search_pipeline_score_calls'] == fields['search_downstream_fit_calls'] == 0

