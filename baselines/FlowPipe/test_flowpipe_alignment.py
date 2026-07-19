from __future__ import annotations

import tempfile
import unittest
import inspect

import numpy as np
import pandas as pd
import torch

from baselines.FlowPipe.agent import GFNAgent
from baselines.FlowPipe.context import SemanticContextLoader
from baselines.FlowPipe.env import FlowPipeEnv
from baselines.FlowPipe.flowpipe import DEFAULT_LLAMA_MODEL_PATH, FlowPipe
from baselines.FlowPipe.model import FiLMLayer, FlowNetwork
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.op_spec import CATALOG


class FlowPipeAlignmentTest(unittest.TestCase):
    def test_flowpipe_exposes_default_llama_checkpoint_root(self):
        parameter = inspect.signature(FlowPipe).parameters["llama_model_path"]
        self.assertEqual(parameter.default, DEFAULT_LLAMA_MODEL_PATH)
        self.assertEqual(DEFAULT_LLAMA_MODEL_PATH, "/mnt/sdb/dengjiale/dppbench/llm_ckpt")

    def test_environment_keeps_cross_task_action_ids_stable(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "target": [0, 1]})

        class Data:
            train_df = frame

        class Evaluator:
            def load_data(self):
                return Data()

        ctx = DataContext(
            task_type="tabular", data_name="toy", primary_table="main",
            tables={"main": TableContext(
                name="main", role="primary", shape=frame.shape,
                dtypes={name: str(frame[name].dtype) for name in frame.columns},
            )},
        )
        env = FlowPipeEnv(Evaluator(), ctx)
        self.assertEqual(env.action_dim, 24 + len(CATALOG) + 1)
        self.assertEqual(int(env.get_action_mask().sum()), 24)

    def test_film_starts_as_identity_modulation(self):
        layer = FiLMLayer(3, 5, 7)
        x = torch.randn(2, 3)
        ctx = torch.randn(2, 7)
        self.assertTrue(torch.allclose(layer(x, ctx), layer.layer(x)))

    def test_flow_network_has_context_conditioned_logz(self):
        model = FlowNetwork(11, 6, hidden_dim=8, ctx_dim=7)
        logits, logz = model(torch.randn(3, 11), torch.randn(3, 7))
        self.assertEqual(tuple(logits.shape), (3, 6))
        self.assertEqual(tuple(logz.shape), (3,))

    def test_state_encoding_matches_official_block_order(self):
        agent = GFNAgent(15, 4, hidden_dim=8, ctx_dim=3)
        state = {
            "logic_pipeline_selected": True, "logic_pipeline_id": 1,
            "stage_index": 0, "predictor_id": 0, "selected_gids": [2],
            "data_feature": np.asarray([9.0, 8.0], dtype=np.float32),
        }
        encoded = agent.encode_state(state, n_logic=2, n_stages=2).cpu().numpy()
        self.assertEqual(encoded.tolist(), [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 9, 8])

    def test_context_loader_uses_contextual_semantic_embedding(self):
        with tempfile.TemporaryDirectory() as root:
            path = f"{root}/Meta-Llama-3.1-8B/contextual_semantic"
            import os
            os.makedirs(path)
            torch.save(torch.arange(4), f"{path}/toy.pt")
            value = SemanticContextLoader(root, "Meta-Llama-3.1-8B", 4).load("toy")
            self.assertEqual(value.tolist(), [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
