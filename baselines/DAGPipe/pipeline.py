"""Self-contained YAML checkpoint; preserves weighted vendor execution."""
from __future__ import annotations

import hashlib
import time
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
        self._transform_signature = None
        self.validate()

    def validate(self):
        if self.payload.get('format') not in {'dppbench.DAGPipe.reference.v2', 'dppbench.DAGPipe.reference.v3', 'dppbench.DAGPipe.reference.v4'}:
            raise ValueError('Unsupported DAGPipe pipeline format; retrain old v1 checkpoints with the corrected v2 adapter')
        for key in ('encoder', 'layers', 'state_dict', 'reference', 'task', 'seed'):
            if key not in self.payload:
                raise ValueError(f'DAGPipe pipeline missing {key}')
        if not self.payload['encoder']:
            raise ValueError('DAGPipe pipeline has no features')
        if self.payload['format'] in ('dppbench.DAGPipe.reference.v3', 'dppbench.DAGPipe.reference.v4'):
            if self.payload.get('scale_protocol') != 'public_v1':
                raise ValueError('DAGPipe v3 checkpoint requires the public_v1 scaling protocol')
            if len(self) > 10:
                raise ValueError('DAGPipe v3 checkpoint exceeds the 10-step budget')
        if self.payload['format'] == 'dppbench.DAGPipe.reference.v4':
            from .protocol_v4 import REFERENCE_ROWS, SPACE, allocate_layers
            import numpy as np
            protocol = self.payload.get('fit_protocol')
            if self.payload.get('operator_space') != SPACE or protocol not in (
                'all_legal_training_rows', 'seeded_training_sample_16', 'seeded_elliptic_training_sample'
            ):
                raise ValueError('Unsupported v4 fit protocol')
            training_rows = self.payload.get('training_rows')
            if not isinstance(training_rows, int) or training_rows < 2:
                raise ValueError('Invalid v4 training row count')
            expected_rows = min(training_rows, REFERENCE_ROWS) if protocol == 'seeded_training_sample_16' else training_rows
            if protocol == 'seeded_elliptic_training_sample':
                expected_rows = self.payload.get('reference_rows')
                if (self.payload['task'] != 'elliptic_bitcoin' or type(expected_rows) is not int
                        or not REFERENCE_ROWS < expected_rows <= training_rows):
                    raise ValueError('Invalid Elliptic reference row count')
            if expected_rows != len(self.payload['reference']):
                raise ValueError('v4 training reference row count mismatch')
            limit = self.payload.get('max_pipeline_len')
            if not isinstance(limit, int) or not 2 <= limit <= 10 or len(self) > limit:
                raise ValueError('Invalid v4 pipeline budget')
            specs = self.payload['encoder']
            if any(s.get('mean') != 0. or s.get('scale') != 1. or s.get('fill') != 0. for s in specs):
                raise ValueError('v4 input encoding cannot hide numeric imputation or scaling')
            ablate_second_robust = self.payload.get('ablate_elliptic_second_robust_scaler', False)
            if not isinstance(ablate_second_robust, bool):
                raise ValueError('Invalid v4 Elliptic ablation flag')
            expected = allocate_layers(specs, np.array(self.payload['reference']),
                                       limit - 1 - len(self.payload.get('adapter', {}).get('ops', [])),
                                       allow_target=self.payload.get('allow_target_encoding', False),
                                       allow_frequency=self.payload.get('allow_frequency_encoding', protocol == 'all_legal_training_rows'),
                                       task=self.payload['task'],
                                       ablate_elliptic_second_robust_scaler=ablate_second_robust)
            if self.payload['layers'] != expected:
                raise ValueError('v4 nodes do not match the declared type/budget allocation')
            target = self.payload.get('category_state', {}).get('TargetEncoder')
            if target and target.get('seed') != self.payload['seed']:
                raise ValueError('TargetEncoder fold seed mismatch')
        elif self.payload.get('operator_space') == 'typed_public_v4':
            raise ValueError('Typed operators require a v4 checkpoint')
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
        identity_started = time.perf_counter()
        signature = self.hash()
        if signature != self._transform_signature:
            # Public payload edits must never retain an old model / cache.
            self.validate()
            self._transforms.clear()
            self._transform_signature = signature
        if device not in self._transforms:
            self._transforms[device] = FrozenTransform(self.payload, device)
        transform = self._transforms[device]
        transform.stats['checkpoint_identity_seconds'] = transform.stats.get('checkpoint_identity_seconds', 0) + time.perf_counter()-identity_started
        return transform.transform(frame)

    def performance_stats(self):
        return {str(device): dict(transform.stats) for device, transform in self._transforms.items()}
