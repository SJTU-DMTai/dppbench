"""Self-contained YAML checkpoint; preserves weighted vendor execution."""
from __future__ import annotations

import hashlib
from pathlib import Path
import yaml


class DAGPipePipeline:
    @staticmethod
    def get_evaluator_class():
        """The checkpoint requires its own validator and replay executor."""
        from .evaluator import DAGPipeEvaluator
        return DAGPipeEvaluator

    def __init__(self, payload):
        self.payload = payload
        self._transforms = {}
        self.validate()

    def validate(self):
        if self.payload.get('format') != 'dppbench.DAGPipe.reference.v1':
            raise ValueError('Unsupported DAGPipe pipeline format')
        for key in ('encoder', 'layers', 'state_dict', 'reference', 'task', 'seed'):
            if key not in self.payload:
                raise ValueError(f'DAGPipe pipeline missing {key}')
        if not self.payload['encoder']:
            raise ValueError('DAGPipe pipeline has no features')
        # Validate names/dimensions, and all required parameters via strict load.
        from .runtime import FrozenTransform
        FrozenTransform(self.payload)

    def op_names(self):
        return self.payload.get('adapter', {}).get('ops', []) + ['DAGPipeInputEncoding'] + [
            name for layer in self.payload['layers'] for name in layer]

    def __len__(self):
        return len(self.op_names())

    def to_yaml(self):
        return yaml.safe_dump(self.payload, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text):
        return cls(yaml.safe_load(text))

    def hash(self):
        return hashlib.sha256(self.to_yaml().encode('utf-8')).hexdigest()

    def save(self, path):
        Path(path).write_text(self.to_yaml(), encoding='utf-8')

    def transform(self, frame, device='cpu'):
        from .runtime import FrozenTransform
        if device not in self._transforms:
            self._transforms[device] = FrozenTransform(self.payload, device)
        return self._transforms[device].transform(frame)
