"""CPU-only regressions for the frozen context encoder; no checkpoint needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from .context import CONTEXT_DIM, GTEEmbedder


class ContextInferenceTest(unittest.TestCase):
    def test_forward_disables_grad_and_returns_trainable_policy_input(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(CONTEXT_DIM))
                self.grad_enabled = None

            def forward(self, attention_mask):
                self.grad_enabled = torch.is_grad_enabled()
                hidden = self.weight.view(1, 1, -1).expand(1, 2, -1) * 2
                return SimpleNamespace(last_hidden_state=hidden)

        embedder = GTEEmbedder.__new__(GTEEmbedder)
        embedder.device = torch.device('cpu')
        embedder._load = Mock()
        embedder._tokenizer = Mock(return_value=Inputs(attention_mask=torch.ones(1, 2)))
        embedder._model = Encoder().eval()
        with torch.enable_grad():
            embedding = embedder.embed_text('synthetic context')
        self.assertFalse(embedder._model.grad_enabled)
        self.assertFalse(embedding.requires_grad)
        self.assertIsNone(embedding.grad_fn)
        self.assertEqual(embedding.shape, (CONTEXT_DIM,))
        torch.testing.assert_close(embedding, torch.full((CONTEXT_DIM,), 2.0))
        policy = torch.nn.Linear(CONTEXT_DIM, 1)
        policy(embedding).sum().backward()
        self.assertIsNotNone(policy.weight.grad)
        self.assertIsNone(embedder._model.weight.grad)


if __name__ == '__main__':
    unittest.main()
