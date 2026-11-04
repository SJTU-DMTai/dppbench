"""Operator loading works even when dppbench.operators is not importable."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize('directory_name', [' operators', 'operators'])
def test_rec_dynamic_operator_directory(tmp_path, directory_name):
    root = Path(__file__).resolve().parents[2]
    source = root / 'dppbench/operators'
    if not source.is_dir():
        source = root / 'dppbench/ operators'
    target = tmp_path / directory_name
    for relative in ('base_op.py', 'integration/join_table.py',
                     'feature_engineering/create_sequence.py'):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    # A fresh process also prevents cached dynamic modules from masking a
    # directory-resolution error in either layout.
    script = r'''
import builtins
from pathlib import Path
import sys
from dppbench.dataset import RecData
from baselines.DAGPipe.task_adapter import rec_splits
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'dppbench.operators' or name.startswith('dppbench.operators.'):
        raise ModuleNotFoundError("No module named 'dppbench.operators'")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
root = Path(sys.argv[1])
data = RecData('fixture')
data.OPERATOR_DIR = str(root / ' operators')
data.data_dir = str(root / 'data')
try:
    rec_splits(data, {'feature': {'target_col': 'rating'}})
except FileNotFoundError as exc:
    assert 'interaction_frozen.parquet' in str(exc), str(exc)
else:
    raise AssertionError('Expected missing frozen fixture')
for suffix in ('integration.join_table', 'feature_engineering.create_sequence'):
    module = sys.modules['_dppbench_ops.' + suffix]
    assert Path(module.__file__).is_relative_to(root)
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
