from __future__ import annotations

import torch

from baselines.common.op_spec import OpCategory
from baselines.DiffPrep.discretizer import featurewise_op_names
from baselines.DiffPrep.search_space import ContinuousPipeline
from baselines.DiffPrep.slot_planner import IDENTITY, Slot
from baselines.DiffPrep.surrogate import TabularSurrogate


def _slot(index: int, candidates: list[str], mandatory: bool = False) -> Slot:
    return Slot(
        index=index,
        category=OpCategory.PREPROCESSING,
        candidates=candidates,
        kind="soft",
        mandatory=mandatory,
        forced_op=candidates[0] if mandatory else None,
    )


def test_official_default_initialization_uses_half_probability() -> None:
    pipeline = ContinuousPipeline(
        [_slot(0, ["ScaleFeature", "TransformPower", IDENTITY])],
        n_features=2,
    )
    beta = pipeline.beta()
    assert torch.allclose(beta[0, :, 2], torch.full((2,), 0.5))
    assert torch.allclose(beta[0, :, :2], torch.full((2, 2), 0.25))


def test_flex_keeps_mandatory_slot_fixed() -> None:
    pipeline = ContinuousPipeline(
        [
            _slot(0, ["HandleMV"], mandatory=True),
            _slot(1, ["ScaleFeature", IDENTITY]),
            _slot(2, ["TransformPower", IDENTITY]),
        ],
        flex=True,
        n_features=2,
    )
    alpha = pipeline.alpha()
    assert torch.allclose(alpha[:, 0, 0], torch.ones(2))
    assert torch.allclose(alpha[:, 0, 1:], torch.zeros(2, 2))
    assert torch.allclose(alpha[:, 1:, 0], torch.zeros(2, 2))


def test_featurewise_argmax_is_not_averaged() -> None:
    pipeline = ContinuousPipeline(
        [_slot(0, ["ScaleFeature", "TransformPower", IDENTITY])],
        n_features=2,
        feature_names=["a", "b"],
    )
    with torch.no_grad():
        pipeline.tau[0, 0, :3] = torch.tensor([5.0, 0.0, 0.0])
        pipeline.tau[0, 1, :3] = torch.tensor([0.0, 5.0, 0.0])
    assert featurewise_op_names(pipeline)[0] == ["ScaleFeature", "TransformPower"]


def test_tabular_surrogate_matches_multiclass_logistic_regression() -> None:
    model = TabularSurrogate(in_dim=4, n_classes=3)
    assert isinstance(model.body, torch.nn.Linear)
    assert model(torch.zeros(5, 4)).shape == (5, 3)
