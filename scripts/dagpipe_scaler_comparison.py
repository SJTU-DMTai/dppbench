"""Read-only local operator comparisons on actual routed DAG inputs.

The old result is an isolated operator evaluation, NOT an old DAG replay.
"""
import numpy as np
import pandas as pd

from dppbench.operators.preprocessing.scale_feature import ScaleFeature
from baselines.dagpipe_old import runtime as old_runtime
from baselines.dagpipe_old.vendor import operations as old_ops


METHODS = {'StandardScaler': 'standard', 'RobustScaler': 'robust', 'MinMaxScaler': 'minmax'}


def compare_input_encoders(payload, training, query=None):
    from baselines.DAGPipe.runtime import InputEncoder
    columns = [spec['column'] for spec in payload['encoder']]
    if any(spec['kind'] != 'numeric' for spec in payload['encoder']):
        raise ValueError('InputEncoder audit currently requires numeric checkpoint columns')
    old = old_runtime.InputEncoder().fit(training[columns])
    current = InputEncoder(preserve_missing=payload.get('operator_space') == 'legacy_daghie').fit(training[columns])
    checkpoint = InputEncoder(payload['encoder'], preserve_missing=current.preserve_missing)
    rows = []
    for split, frame in [('training', training), ('query', query)]:
        if frame is None:
            continue
        a, b, c = old.transform(frame), current.transform(frame), checkpoint.transform(frame)
        for i, column in enumerate(columns):
            row = dict(split=split, column=column, rows=len(frame))
            for label, x, y in [('old_current', a[:, i], b[:, i]), ('refit_checkpoint', b[:, i], c[:, i])]:
                finite = np.isfinite(x) & np.isfinite(y)
                row[label + '_max_abs_diff'] = float(np.abs(x[finite] - y[finite]).max()) if finite.any() else None
                row[label + '_mismatches'] = int((~np.isclose(x, y, rtol=1e-10, atol=1e-12, equal_nan=True)).sum())
            rows.append(row)
    return rows


class ScalerComparison:
    def __init__(self, payload, reference_rows):
        self.layers = payload['layers']
        self.columns = [spec['column'] for spec in payload['encoder']]
        self.n_ref = reference_rows
        self.inputs, self.fitted, self.rows = {}, {}, {}

    def observe(self, stage, value, layer=None, node=None, **details):
        if node is None or self.layers[layer][node] not in METHODS:
            return
        key = (layer, node)
        name = self.layers[layer][node]
        if stage == 'after_routing':
            self.inputs[key] = value.detach()
        elif stage == 'operator_output':
            x = self.inputs.pop(key)
            array = x.cpu().numpy()
            if key not in self.fitted:
                public = ScaleFeature(cols=self.columns, method=METHODS[name])
                public.transform(pd.DataFrame(array[:self.n_ref], columns=self.columns))
                self.fitted[key] = public
            public = self.fitted[key]
            expected = public.transform(pd.DataFrame(array[self.n_ref:], columns=self.columns)).to_numpy()
            actual = value[self.n_ref:].detach().cpu().numpy()
            old = old_runtime.reference_operation(getattr(old_ops, name)(), x, self.n_ref)[self.n_ref:].cpu().numpy()
            ref = array[:self.n_ref]
            for i, column in enumerate(self.columns):
                row_key = (*key, column)
                row = self.rows.setdefault(row_key, dict(layer=layer, node=node, op=name, column=column,
                    rows=0, public_reference_params=list(map(float, public.params_[column])),
                    old_reference_median=float(np.sort(ref[:, i])[(len(ref)-1)//2]),
                    old_reference_iqr=float(np.percentile(ref[:, i], 75)-np.percentile(ref[:, i], 25)),
                    current_abs_max=0., public_abs_max=0., old_local_abs_max=0.,
                    current_public_max_abs_diff=0., current_old_max_abs_diff=0.,
                    current_public_mismatches=0, current_old_mismatches=0, nonfinite_values=0))
                row['rows'] += len(actual)
                for label, values in [('current', actual[:, i]), ('public', expected[:, i]), ('old_local', old[:, i])]:
                    finite = values[np.isfinite(values)]
                    row['nonfinite_values'] += int((~np.isfinite(values)).sum())
                    if finite.size:
                        row[label + '_abs_max'] = max(row[label + '_abs_max'], float(np.abs(finite).max()))
                for label, other in [('public', expected[:, i]), ('old', old[:, i])]:
                    diff = np.abs(actual[:, i] - other)
                    finite = diff[np.isfinite(diff)]
                    if finite.size:
                        field = 'current_' + label + '_max_abs_diff'
                        row[field] = max(row[field], float(finite.max()))
                    # Public output is float32; preserve its real dtype and allow rounding.
                    row['current_' + label + '_mismatches'] += int((~np.isclose(
                        actual[:, i], other, rtol=2e-6, atol=1e-6, equal_nan=True)).sum())

    def result(self):
        return list(self.rows.values())
