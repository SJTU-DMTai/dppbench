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
    def test_single_task_training_collects_candidates_and_updates(self):
        from unittest.mock import Mock
        from baselines.FlowPipe.trainer import FlowPipeTrainer
        from baselines.common.pipeline import Pipeline
        env = Mock()
        env.current_pipeline.return_value = Pipeline()
        env.logic_id = 0
        agent = Mock()
        agent.sample_trajectory.return_value = (0.7, {'actions': [0], 'errors': []})
        agent.learn_from_batch.return_value = [0.1]
        with tempfile.TemporaryDirectory() as root:
            trainer = FlowPipeTrainer(agent, [(env, torch.zeros(4))], output_dir=root,
                combos_per_epoch=1, trajectories_per_combo=2, updates_per_combo=1,
                trajectory_batch_size=2)
            history = trainer.train(num_epochs=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(trainer.candidates), 4)
        self.assertEqual(trainer.optimizer_updates, 2)
        self.assertEqual(agent.sample_trajectory.call_count, 4)
        self.assertEqual(agent.learn_from_batch.call_count, 2)

    def test_flowpipe_exposes_default_llama_checkpoint_root(self):
        parameter = inspect.signature(FlowPipe).parameters["llama_model_path"]
        self.assertEqual(parameter.default, DEFAULT_LLAMA_MODEL_PATH)
        self.assertEqual(DEFAULT_LLAMA_MODEL_PATH, "/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B")

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

    def test_mandatory_masks_and_rec_prefix_execution(self):
        from baselines.common.op_spec import OpCategory
        from baselines.common.pipeline import check_pipeline
        frame = pd.DataFrame({'user_id': [1, 1], 'item_id': [2, 3],
                              'timestamp': [1, 2], 'rating': [0, 1]})
        from types import SimpleNamespace
        ctx = DataContext(task_type='rec', data_name='toy', primary_table='interaction',
            secondary_tables=['item'],
            tables={
                'interaction': TableContext(name='interaction', role='primary', shape=frame.shape,
                    dtypes={c: str(frame[c].dtype) for c in frame}),
                'item': TableContext(name='item', role='secondary', shape=(2, 1),
                    dtypes={'item_id': 'int64'}, join_keys_to_primary=[{'primary':'item_id','secondary':'item_id'}]),
            })

        class Evaluator:
            def load_data(self):
                return SimpleNamespace(interaction_df=frame)

            def run_preprocessing(self, pipeline):
                report = check_pipeline(pipeline, ctx, require_mandatory=False)
                if not report.valid:
                    raise ValueError(str(report.errors))
                processed = frame.copy()
                processed['item_id_seq'] = [[], [2]]
                return processed

        env = FlowPipeEnv(Evaluator(), ctx, max_steps=2)
        mask = env.get_action_mask()
        for i in np.flatnonzero(mask):
            self.assertEqual(set(env.logical_pipelines[i][:2]),
                             {OpCategory.INTEGRATION, OpCategory.FEATURE_ENGINEERING})
        order_id = next(i for i in np.flatnonzero(mask)
                        if env.logical_pipelines[i][0] == OpCategory.FEATURE_ENGINEERING)
        env.step(int(order_id))
        action_mask = env.get_action_mask()
        self.assertEqual(action_mask[env.skip_gid], 0)
        sequence_gid = next(a.gid for a in env.actions if a.name == 'CreateSequence')
        self.assertEqual(list(np.flatnonzero(action_mask)), [sequence_gid])
        _, _, done, info = env.step(sequence_gid)
        self.assertFalse(done, info)
        self.assertEqual(info['err_code'], 0)
        self.assertFalse(check_pipeline(env.current_pipeline(), ctx).valid)
        join_gid = next(a.gid for a in env.actions if a.name == 'JoinTable')
        self.assertEqual(list(np.flatnonzero(env.get_action_mask())), [join_gid])

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
