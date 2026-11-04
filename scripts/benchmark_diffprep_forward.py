"""Synthetic-only DiffPrep forward cost. No search, HPO, model fit or data download.
Run from DPPBench root with an importable test_tabular_correctness_fixes module.
"""
import argparse
import json
import os
import time
import tracemalloc
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from test_tabular_correctness_fixes import TabularFixTests
from baselines.common.data_ctx import infer_data_context
from baselines.common.op_spec import OpCategory
from baselines.DiffPrep.slot_planner import Slot
from baselines.DiffPrep.tabular_bank import TabularTransformBank
from baselines.DiffPrep.search_space import ContinuousPipeline


def run(n, repeats):
    case = TabularFixTests()
    f = pd.DataFrame({'id': range(n), 'x': np.arange(n, dtype=float) % 13,
                      'cat': ['c'+str(i % 7) for i in range(n)],
                      'y': np.arange(n) % 2})
    d = case.data()
    train_count = max(2, int(n * 0.7))
    d.train_df = f.iloc[:train_count].copy()
    ctx = infer_data_context('synthetic', d, task_type='tabular')
    slot = Slot(0, OpCategory.PREPROCESSING, ['OneHotEncode', 'Identity'], 'soft')
    bank = TabularTransformBank(d, ctx, [slot], f, np.arange(train_count), ['x', 'cat'], 43)
    cont = ContinuousPipeline([slot], n_features=2, feature_names=bank.names)
    cont.tabular_bank = bank
    x, mask = bank.inputs('cpu')
    op = bank._candidate
    calls = [0]
    def wrapped(*args, **kwargs):
        calls[0] += 1
        return op(*args, **kwargs)
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[0]
    times=[]
    with patch.object(bank, '_candidate', side_effect=wrapped):
        for _ in range(repeats):
            t=time.perf_counter()
            with torch.no_grad():
                output=cont(x, mask)
            times.append(time.perf_counter()-t)
    current, peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'rows':n,'fixed_training_reference_rows':len(bank.train_rows),
            'input_features':len(bank.names),'output_width':bank.output_dim,
            'output_tensor_mib':round(output.numel()*output.element_size()/2**20,4),
            'repeats':repeats,'candidate_calls':calls[0],
            'mean_seconds':round(float(np.mean(times)),6),
            'max_seconds':round(max(times),6),
            'tracemalloc_peak_extra_mib':round(max(0,peak-start_mem)/2**20,4),
            'caveat':'tracemalloc excludes native allocator and torch RSS peaks'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--rows', nargs='+', type=int, default=[30,100])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', default='diffprep_forward_benchmark.json')
    a=p.parse_args()
    results=[run(n,a.repeats) for n in a.rows]
    with open(a.output,'w',encoding='utf-8') as f:json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))
