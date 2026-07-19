from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.HAIPipe.active_learning import pipeline_features
from baselines.HAIPipe.combiner import _subsequences
from baselines.HAIPipe.features import dataset_features


def test_dataset_features_match_upstream_shape_and_order():
    frame = pd.DataFrame({"number": [1.0, 2.0, np.nan], "text": ["a", "a", "b"]})
    features = dataset_features(frame)
    assert features.shape == (100 * 19,)
    assert features[1] == 3
    assert features[2] == 1 / 3
    assert features[3] == 1
    assert features[19 + 3] == 2


def test_all_nonempty_ordered_subsequences_are_enumerated():
    steps = [PipelineStep("a"), PipelineStep("b"), PipelineStep("c")]
    names = [tuple(step.op for step in seq) for seq in _subsequences(steps)]
    assert len(names) == 7
    assert ("a", "c") in names
    assert ("c", "a") not in names


def test_pipeline_features_use_checkpoint_vocabulary():
    pipe = Pipeline([PipelineStep("b"), PipelineStep("unknown")])
    encoded = pipeline_features(pipe, ["a", "b"], sequence_length=4)
    assert encoded.tolist() == [2, 0, 0, 0]
