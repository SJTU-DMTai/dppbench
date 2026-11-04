"""Real dataframe forwards for the existing feature-wise DiffPrep relaxation.

Each candidate is fitted on the fixed training reference, not the loss batch.
Real expanded outputs occupy separate blocks. Architecture gates and centered
numerical input derivatives remain in torch; no candidate is replaced by rank.
"""
import copy
import random
import numpy as np
import pandas as pd
import torch

from .slot_planner import IDENTITY, diffprep_make_step


class TabularTransformBank:
    version = 'tabular_real_ops_v1'

    def __init__(self, data, ctx, slots, frame, train_rows, feature_names, seed):
        self.data, self.ctx, self.slots = data, ctx, slots
        self.names = list(feature_names)
        self.train_rows = np.asarray(train_rows, dtype=int)
        self.frame = frame.reset_index(drop=True).copy()
        self.categories = {}
        self.raw_kinds = {}
        self.params = {}
        self.blocks = {}
        self.output_dim = len(self.names)
        self.last_columns = {}
        self.last_retained = {}
        self.position_columns = {}
        self.position_retained = {}
        self.last_missing = {}
        rng = random.Random(seed)
        for c in self.names:
            if pd.api.types.is_datetime64_any_dtype(frame[c]):
                raise ValueError('Datetime columns belong to immutable model context, not numerical operator choices')
            if not pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_datetime64_any_dtype(frame[c]):
                self.categories[c] = list(pd.unique(frame.iloc[self.train_rows][c].dropna()))
            self.raw_kinds[c] = 'category' if c in self.categories else 'numeric'
        for i, slot in enumerate(slots):
            for name in slot.candidates:
                if name == IDENTITY:
                    continue
                step = diffprep_make_step(name, ctx, rng)
                if step is None:
                    raise ValueError(f'DiffPrep cannot materialize existing candidate {name}')
                self.params[i, name] = copy.deepcopy(step.params)
                # Mandatory alternatives have exactly zero probability. Keep
                # their candidates/logits, without allocating unreachable blocks.
                if slot.forced_op and name != slot.forced_op:
                    continue
                for j, c in enumerate(self.names):
                    width = len(self.train_rows) if name == 'OneHotEncode' else 1 if name in ('FrequencyEncode', 'HashEncode', 'TargetEncode') else 0
                    if width:
                        self.blocks[i, name, j] = slice(self.output_dim, self.output_dim + width)
                        self.output_dim += width
        self.transform_width = self.output_dim
        # Keep original immutable time coordinates in the model input, as the
        # real evaluator does, without adding them to the learned search space.
        self.context_columns = [c for c in frame if c not in self.names and c not in (ctx.target_col, ctx.id_col)
                                and (c in (ctx.time_col, getattr(data, '_sort_col', None))
                                     or pd.api.types.is_datetime64_any_dtype(frame[c]))]
        self.output_dim += len(self.context_columns)
        from dppbench.memory import require_memory_headroom
        require_memory_headroom(len(frame) * self.output_dim * 8 * 4, 'DiffPrep real operator blocks')

    def inputs(self, device):
        columns = []
        for c in self.names:
            s = self.frame[c]
            if c in self.categories:
                mapping = {v: i for i, v in enumerate(self.categories[c])}
                values = s.map(mapping).fillna(-1).mask(s.isna()).astype(float)
            elif pd.api.types.is_datetime64_any_dtype(s):
                # Same numeric view used by the final LightGBM adapter.
                values = pd.to_datetime(s).astype('int64') / 1e9
            else:
                values = pd.to_numeric(s, errors='raise')
            columns.append(values.to_numpy(dtype=float))
        arr = np.column_stack(columns)
        if np.isinf(arr).any():
            raise ValueError('Infinite raw features require an explicit data policy; refusing implicit replacement')
        mask = np.isnan(arr)
        values = np.nan_to_num(arr, nan=0., posinf=0., neginf=0.)
        return (torch.tensor(values, dtype=torch.float64, device=device),
                torch.tensor(mask, dtype=torch.bool, device=device))

    def _params(self, i, name, c, categorical):
        params = copy.deepcopy(self.params[i, name])
        # The original feature-wise projection chooses source columns. All
        # other method/threshold/smoothing parameters are frozen and exported.
        if name == 'DiscretizeFeature':
            bounds = params.get('boundaries', {})
            if bounds:
                params['boundaries'] = {c: next(iter(bounds.values()))}
        elif name == 'LabelEncode':
            params['cols'] = [c] if categorical else []
        else:
            params['cols'] = [c]
        return params

    def _series(self, x, j, categorical, missing):
        values = x.detach().cpu().numpy()[:, j].copy()
        c = self.names[j]
        if categorical:
            cats = self.categories.get(c, [])
            values = np.array([cats[int(v)] if np.isfinite(v) and 0 <= int(v) < len(cats)
                               else self.frame[c].iloc[r] for r, v in enumerate(values)], dtype=object)
        s = pd.Series(values, name=c)
        if missing is not None:
            s = s.mask(missing.detach().cpu().numpy()[:, j])
        return s

    def _numeric(self, frame, c):
        out = []
        for col in frame:
            s = frame[col]
            if pd.api.types.is_numeric_dtype(s):
                a = s.to_numpy(dtype=float)
            else:
                # Raw categorical input view only; real LabelEncode still runs
                # at its learned/mandatory position with its own training map.
                cats = self.categories.setdefault(c, [])
                for v in pd.unique(s.iloc[self.train_rows].dropna()):
                    if v not in cats:
                        cats.append(v)
                mapping = {v: i for i, v in enumerate(cats)}
                a = s.map(mapping).fillna(-1).to_numpy(dtype=float)
            out.append(np.nan_to_num(a, nan=0., posinf=0., neginf=0.))
            if np.isinf(a).any():
                raise ValueError('Real operator produced infinite values; refusing a different surrogate transform')
        return np.column_stack(out) if out else np.empty((len(frame), 0))

    def _candidate(self, i, name, j, x, categorical, missing):
        c = self.names[j]
        params = self._params(i, name, c, categorical)
        op = self.data._load_op(name)(**self.data._resolve_param(params))
        source = self._series(x, j, categorical, missing)
        full = source.to_frame()
        target = self.ctx.target_col
        fit = full.iloc[self.train_rows].copy()
        if name == 'TargetEncode':
            fit[target] = self.frame.iloc[self.train_rows][target].to_numpy()
        op.transform(fit)
        result = op.transform(full)
        result = result.drop(columns=[target], errors='ignore')
        if len(result) != len(full) or not result.columns.is_unique:
            raise ValueError(f'DiffPrep {name} must preserve query rows and unique output columns')
        retained = c in result
        self.last_missing[i, name, j] = (result[c].isna().to_numpy() if retained else np.zeros(len(full), dtype=bool))
        self.last_retained[i, name, j] = retained
        self.position_retained[self._position, i, name, j] = retained
        base = self._numeric(result[[c]], c)[:, 0] if retained else np.zeros(len(full))
        extra = result.drop(columns=[c], errors='ignore')
        values = self._numeric(extra, c)
        self.last_columns[i, name, j] = list(extra)
        self.position_columns[self._position, i, name, j] = list(extra)
        # Differentiate the REAL fitted transform with a fixed fit reference.
        derivative_base = np.zeros(len(full))
        derivative_extra = np.zeros_like(values)
        if not categorical:
            eps = 2e-4 if name == 'ScaleFeature' else 1e-6
            perturbed = []
            for sign in (1, -1):
                query = full.copy()
                query[c] = query[c] + sign * eps
                p = op.transform(query).drop(columns=[target], errors='ignore')
                # OHE uses its fitted output schema. No query-specific columns.
                p = p.reindex(columns=result.columns)
                perturbed.append((self._numeric(p[[c]], c)[:, 0] if retained else np.zeros(len(full)), self._numeric(p.drop(columns=[c], errors='ignore'), c)))
            derivative_base = (perturbed[0][0] - perturbed[1][0]) / (2 * eps)
            derivative_extra = (perturbed[0][1] - perturbed[1][1]) / (2 * eps)
        return base, values, derivative_base, derivative_extra

    def forward(self, x, mask, beta, alpha=None):
        if len(x) != len(self.frame):
            raise ValueError('DiffPrep tabular bank requires the full fixed reference/query view before minibatch selection')
        out = torch.cat([x, x.new_zeros((len(x), self.transform_width - len(self.names)))], dim=1)
        categorical = dict(self.raw_kinds)
        missing = mask
        present = {c: True for c in self.names}

        def apply(i, source):
            slot = self.slots[i]
            branches = []
            for k, name in enumerate(slot.candidates):
                if slot.forced_op and name != slot.forced_op:
                    continue
                parts = []
                # Every source feature has its own gate. Existing generated
                # blocks pass through; appended outputs have genuine width.
                block_values = {}
                for j, c in enumerate(self.names):
                    weight = beta[i, j, k]
                    if name == IDENTITY or not present[c]:
                        base, extra = source[:, j], None
                    else:
                        a, b, da, db = self._candidate(i, name, j, source, categorical[c] == 'category', missing)
                        origin = source[:, j]
                        base = source.new_tensor(a) + source.new_tensor(da) * (origin - origin.detach())
                        extra = source.new_tensor(b) + source.new_tensor(db) * (origin - origin.detach())[:, None]
                    parts.append(weight * base)
                    block = self.blocks.get((i, name, j))
                    if block is not None and extra is not None:
                        if extra.shape[1] > block.stop - block.start:
                            raise ValueError('DiffPrep expanded operator exceeded its declared block; no truncation allowed')
                        block_values[i, name, j] = torch.cat([extra, source.new_zeros(
                            (len(source), block.stop-block.start-extra.shape[1]))], dim=1)
                    # Weights for the feature's generated blocks follow the
                    # original source gate, rather than new architecture logits.
                expanded = [block_values.get(key, source[:, sl]) * beta[i, key[2], k]
                            for key, sl in self.blocks.items()]
                branch = torch.cat([torch.stack(parts, dim=1), *expanded], dim=1)
                branches.append(branch)
            return torch.stack(branches).sum(0)

        def slot_missing(i, previous):
            masks = []
            for j, c in enumerate(self.names):
                flags = []
                for k, name in enumerate(self.slots[i].candidates):
                    if float(beta[i, j, k].detach()) > 0.:
                        flags.append(previous[:, j] if name == IDENTITY or not present[c]
                                     else x.new_tensor(self.last_missing[i, name, j], dtype=torch.bool))
                masks.append(torch.stack(flags).all(0))
            return torch.stack(masks, dim=1)

        if alpha is None:
            for i, slot in enumerate(self.slots):
                self._position = i
                out = apply(i, out)
                missing = slot_missing(i, missing)
                if slot.forced_op == 'LabelEncode':
                    categorical = {c: 'numeric' for c in self.names}
                for j, c in enumerate(self.names):
                    weights = beta[i, j, :slot.n_candidates].detach()
                    if float(weights.max()) == 1.:
                        name = slot.candidates[int(weights.argmax())]
                        if name != IDENTITY and present[c]:
                            present[c] = self.last_retained.get((i, name, j), True)
        else:
            # Flex keeps mandatory input stages fixed; optional-stage ordering
            # remains the existing per-feature Sinkhorn relaxation.
            for pos in range(len(self.slots)):
                self._position = pos
                outputs = [apply(i, out) for i in range(len(self.slots))]
                stage_masks = [slot_missing(i, missing) for i in range(len(self.slots))]
                weights = []
                for i, value in enumerate(outputs):
                    owner = list(range(len(self.names))) + [j for (_, _, j), sl in self.blocks.items() for _ in range(sl.stop-sl.start)]
                    weights.append(value * alpha[owner, pos, i][None, :])
                out = sum(weights)
                missing = torch.stack([torch.stack([stage_masks[i][:, j]
                    for i in range(len(self.slots)) if float(alpha[j, pos, i].detach()) > 0.]).all(0)
                    for j in range(len(self.names))], dim=1)
                if self.slots[pos].forced_op == 'LabelEncode':
                    categorical = {c: 'numeric' for c in self.names}
                for j, c in enumerate(self.names):
                    aw = alpha[j, pos].detach()
                    if float(aw.max()) == 1.:
                        i = int(aw.argmax())
                        bw = beta[i, j, :self.slots[i].n_candidates].detach()
                        if float(bw.max()) == 1.:
                            name = self.slots[i].candidates[int(bw.argmax())]
                            if name != IDENTITY and present[c]:
                                present[c] = self.position_retained.get((pos, i, name, j), True)
        self.output_missing = missing.detach().cpu().numpy()
        context = []
        for c in self.context_columns:
            s = self.frame[c]
            from dppbench.tabular_contracts import datetime_seconds
            values = datetime_seconds(s) if pd.api.types.is_datetime64_any_dtype(s) else pd.to_numeric(s, errors='raise')
            context.append(out.new_tensor(values.to_numpy(dtype=float))[:, None])
        return torch.cat([out, *context], dim=1) if context else out

    def projected_frame(self, continuous, output):
        """Decode an exactly one-hot architecture for real-export acceptance."""
        from .discretizer import featurewise_op_names, slot_order
        choices = featurewise_op_names(continuous)
        cols = {c: output[:, j].detach().cpu().numpy() for j, c in enumerate(self.names)}
        for j, c in enumerate(self.names):
            cols[c] = cols[c].copy()
            cols[c][self.output_missing[:, j]] = np.nan
        present = set(self.names)
        orders = self.feature_orders(continuous)
        for pos in range(len(self.slots)):
            for j, c in enumerate(self.names):
                i = orders[j][pos]
                name = choices[i][j]
                if name == IDENTITY or c not in present:
                    continue
                if not self.position_retained.get((pos if continuous.flex else i, i, name, j), True):
                    cols.pop(c, None)
                    present.discard(c)
                sl = self.blocks.get((i, name, j))
                if sl is not None:
                    for k, col in enumerate(self.position_columns.get((pos if continuous.flex else i, i, name, j), [])):
                        cols[col] = output[:, sl.start+k].detach().cpu().numpy()
        for k, c in enumerate(self.context_columns):
            cols[c] = output[:, self.transform_width+k].detach().cpu().numpy()
        return pd.DataFrame(cols)

    def feature_orders(self, continuous):
        if not continuous.flex:
            return [list(range(len(self.slots))) for _ in self.names]
        alpha = continuous.alpha().detach().cpu()
        orders = []
        for j in range(len(self.names)):
            remaining = set(range(len(self.slots)))
            order = []
            for pos in range(len(self.slots)):
                i = max(sorted(remaining), key=lambda k: float(alpha[j, pos, k]))
                order.append(i)
                remaining.remove(i)
            orders.append(order)
        return orders

    def selected_steps(self, continuous):
        import json
        from baselines.common.pipeline import PipelineStep
        from .discretizer import featurewise_op_names, slot_order
        choices = featurewise_op_names(continuous)
        steps = []
        kinds = dict(self.raw_kinds)
        orders = self.feature_orders(continuous)
        for pos in range(len(self.slots)):
            groups = {}
            for j, c in enumerate(self.names):
                i = orders[j][pos]
                name = choices[i][j]
                if name != IDENTITY:
                    params = self._params(i, name, c, kinds[c] == 'category')
                    # Real scalar-column transforms are separable except row
                    # l2 normalization. Keep l2 steps separate to match forward.
                    signature = (i, name, json.dumps({k: v for k, v in params.items() if k not in ('cols', 'boundaries')}, sort_keys=True), j if params.get('method') == 'l2' else None)
                    if signature not in groups:
                        groups[signature] = PipelineStep(name, params=params)
                    else:
                        group = groups[signature].params
                        group['cols'] = list(dict.fromkeys(group.get('cols', []) + params.get('cols', [])))
                        if 'boundaries' in params:
                            group.setdefault('boundaries', {}).update(params['boundaries'])
                    if name == 'LabelEncode':
                        kinds[c] = 'numeric'
            steps.extend(groups.values())
        return steps

    def contract(self):
        from dppbench.tabular_contracts import frame_hash
        return {'version': self.version, 'train_rows': len(self.train_rows),
                'reference': 'fixed_training_partition', 'feature_columns': self.names,
                'training_reference_sha256': frame_hash(self.frame.iloc[self.train_rows]),
                'surrogate_width': self.output_dim,
                'candidate_params': {f'{i}:{n}': p for (i, n), p in self.params.items()},
                'expanded_outputs': True, 'query_labels_used_for_fit': False}
