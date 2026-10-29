"""Exact recommendation transforms with a differentiable mixture over their outputs.

Variable-width encoders cannot be replaced by scalar bucketizers. Each original
feature keeps DiffPrep's per-slot logits. We evaluate the real train-fitted
operator compositions, embed their output columns in a common zero-padded space,
and mix paths with the product of the original slot probabilities. Argmax exports
the same real operators/columns. Bilevel losses/optimizers/budgets are unchanged.
"""
from itertools import product, permutations
from pathlib import Path
import copy
import random
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from dppbench.rec_protocol import assert_disjoint, training_rows, VERSION
from baselines.common.data_ctx import infer_data_context
from baselines.common.pipeline import PipelineStep, check_pipeline, Pipeline
from .slot_planner import diffprep_make_step, IDENTITY


def prepare_rec_search(data, config, seed, small_n):
    """Audit projected test identities, never test labels, before optimization."""
    directory = Path(data._std_test_dir(data.data_dir))
    train_path, test_path = directory/'interaction_frozen.parquet', directory/'std_test.parquet'
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError('DiffPrep rec requires frozen training and test identity artifacts')
    frame = pd.read_parquet(train_path)
    if '__split__' in frame and not frame['__split__'].eq('train').all():
        raise ValueError('Frozen training file contains held-out partitions')
    time = config.get('feature', {}).get('time_col', 'timestamp')
    identity_cols = [data._user_id_col, data._item_id_col, time]
    identities = pd.read_parquet(test_path, columns=identity_cols)
    audit = assert_disjoint(frame, identities, *identity_cols)
    data.model_cfg = copy.deepcopy(config)
    data.model_cfg.setdefault('train', {})['seed'] = seed
    data.interaction_df = frame.assign(__split__='train')
    data._validation_prepared = False
    data._prepare_validation()
    train = training_rows(data.interaction_df)
    val = data.interaction_df.loc[data.interaction_df['__split__'].eq('val')]
    # Context is the full legal training partition, NOT the optimizer query sample.
    data._rec_history_reference = train[identity_cols].assign(__split__='train').copy()
    if small_n and len(train) > small_n:
        # Keep the established recent-history sampling strategy, within train only.
        train = train.sort_values(time, kind='stable').tail(small_n)
    data.interaction_df = pd.concat([train, val], ignore_index=True)
    # Same fixed JoinTable adapter that will be exported, before architecture search.
    structural = []
    sides = {}
    for name in ('user', 'item'):
        if getattr(data, name+'_df', None) is not None:
            sides[name+'_df'] = '$'+name+'_df'
    if sides:
        params = dict(method='rec', user_col=data._user_id_col, item_col=data._item_id_col, how='left', **sides)
        op = data._load_op('JoinTable')(**data._resolve_param(params))
        data.interaction_df = data._transform_rec_dag_df(data.interaction_df, op, 'JoinTable')
        structural.append(PipelineStep('JoinTable', params=params))
    seq_params = dict(user_col=data._user_id_col, item_col=data._item_id_col, time_col=time,
                      seq_col='item_id_seq', max_len=config.get('feature', {}).get('seq_max_len', 20))
    op = data._load_op('CreateSequence')(**seq_params)
    data.interaction_df = op.transform_with_history(data.interaction_df, data._rec_history_reference)
    data.register_col_types(op.output_col_types)
    data._diffprep_sequence_step = PipelineStep('CreateSequence', params=seq_params)
    ctx = infer_data_context(data.name, data, task_type='rec')
    audit.update(protocol=VERSION, train_rows=len(train), val_rows=len(val),
                 history_protocol='full_training_context_query_subsample_v1',
                 history_reference_rows=len(data._rec_history_reference),
                 history_reference_sha256=hashlib.sha256(pd.util.hash_pandas_object(
                     data._rec_history_reference[identity_cols].astype('string'), index=False).to_numpy().tobytes()).hexdigest(),
                 validation_ratio=config.get('train', {}).get('val_ratio', .1),
                 structural_join_params=[s.params for s in structural])
    return ctx, structural, audit


def fixed_rec_list_inputs(data, device='cpu'):
    """Use the same mapped IDs, clipping, padding and explicit lengths as DIN.

    Called only after train-fitted remapping. Labels do not build representations.
    The primary item history has its own shared item-embedding input.
    """
    from dppbench.models.inputs import build_feature_columns, VarLenSparseFeat, pad_sequences
    frame = data.interaction_df
    columns = build_feature_columns(training_rows(frame), data.model_cfg, data.col_types)
    specs, inputs = {}, {}
    for fc in columns:
        if not isinstance(fc, VarLenSparseFeat) or fc.name == 'item_id_seq':
            continue
        ids, lengths = pad_sequences(frame[fc.name].tolist(), fc.maxlen)
        ids[(ids < 0) | (ids >= fc.vocabulary_size)] = 0
        ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        mask = torch.arange(fc.maxlen, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        source = ('user' if fc.embedding_name == data._user_id_col else
                  'item' if fc.embedding_name == data._item_id_col else 'own')
        specs[fc.name] = {'vocabulary_size': fc.vocabulary_size, 'embedding_source': source,
                         'maxlen': fc.maxlen}
        inputs[fc.name] = {'ids': ids, 'lengths': lengths, 'mask': mask}
    return specs, inputs


class RecTransformBank:
    """No new candidates: exact compositions of the existing eligible slots.

    Sparse matrices avoid materializing every zero-padded OneHot block. Nothing
    is pruned or evaluated with a different objective to save computation.
    """
    def __init__(self, data, ctx, slots, seed, flex=False, max_features=None):
        self.frame = data.interaction_df.reset_index(drop=True).copy()
        self.slots = slots
        self.feature_names = [c for c in self.frame if c not in {
            ctx.target_col, ctx.user_col, ctx.item_col, ctx.time_col, '__split__', 'sample_weight',
            f'{ctx.target_col}_corrected'} and data.col_types.get(c) in ('numeric', 'categorical')]
        if max_features:
            self.feature_names = self.feature_names[:max_features]
        if not self.feature_names:
            raise ValueError('DiffPrep rec has no deployable side features; dummy feature search is forbidden')
        self.learned_slots = [i for i, s in enumerate(slots) if s.kind == 'soft']
        reorderable = [i for i, s in enumerate(slots) if not s.mandatory]
        self.orders = []
        for perm in (permutations(reorderable) if flex else [reorderable]):
            order = list(range(len(slots)))
            for pos, slot_index in zip(reorderable, perm):
                order[pos] = slot_index
            self.orders.append(tuple(order))
        self.paths = list(product(*(range(slots[i].n_candidates) for i in self.learned_slots)))
        if not self.paths:
            self.paths = [()]
        self.paths = [(order, path) for order in self.orders for path in self.paths]
        self.blocks, self.columns, self.step_params = {}, {}, {}
        self.feature_paths, self.union_columns = {}, {}
        self.valid_mask = np.full((len(slots), len(self.feature_names), max(s.n_candidates for s in slots)), -np.inf, dtype=np.float32)
        numeric_ops = {'ScaleFeature', 'TransformPower', 'ClipOutlier', 'DiscretizeFeature'}
        category_ops = {'OneHotEncode', 'HashEncode', 'TargetEncode', 'FrequencyEncode', 'LabelEncode', 'OrdinalEncode'}
        for i, slot in enumerate(slots):
            for f, name in enumerate(self.feature_names):
                kind = data.col_types[name]
                for j, op in enumerate(slot.candidates):
                    eligible = not ((op in numeric_ops and kind != 'numeric') or
                                    (op in category_ops and kind != 'categorical'))
                    if eligible:
                        self.valid_mask[i, f, j] = 0
        self.widths = []
        for feature_index, feature in enumerate(self.feature_names):
            prepared = []
            union = []
            self.feature_paths[feature_index] = []
            for path in self.paths:
                if any(not np.isfinite(self.valid_mask[i, feature_index, j])
                       for i, j in zip(self.learned_slots, path[1])):
                    continue
                self.feature_paths[feature_index].append(path)
                frame = self.frame.copy()
                d = copy.copy(data)
                d.col_types = dict(data.col_types)
                d.model_cfg = copy.deepcopy(data.model_cfg)
                # Remapping is in-place; candidate side tables must not alias.
                for attr in ('user_df', 'item_df', 'std_test_negatives_df'):
                    side = getattr(data, attr, None)
                    if side is not None:
                        setattr(d, attr, side.copy(deep=True))
                steps = []
                order, choices = path
                choice_by_slot = dict(zip(self.learned_slots, choices))
                for slot_index in order:
                    if slot_index not in choice_by_slot:
                        continue
                    candidate = choice_by_slot[slot_index]
                    op_name = slots[slot_index].candidates[candidate]
                    if op_name == IDENTITY:
                        continue
                    step = diffprep_make_step(op_name, ctx, random.Random(seed))
                    if step is None:
                        raise ValueError(f'DiffPrep cannot materialize candidate {op_name}')
                    import inspect
                    if 'cols' in inspect.signature(d._load_op(op_name).__init__).parameters:
                        step.params['cols'] = [feature]
                    elif 'boundaries' in step.params:
                        step.params['boundaries'] = {feature: next(iter(step.params['boundaries'].values()))}
                    else:
                        raise ValueError(f'Recommendation soft candidate {op_name} has no feature selector')
                    errors = check_pipeline(Pipeline([step]), ctx, require_mandatory=False)
                    if not errors.valid:
                        raise ValueError('; '.join(errors.errors))
                    op = d._load_op(op_name)(**d._resolve_param(step.params))
                    frame = d._transform_rec_dag_df(frame, op, op_name)
                    steps.append(step)
                # Identify actual outputs owned by this original feature.
                original_other = set(self.frame.columns) - {feature}
                outputs = [c for c in frame if c not in original_other and d.col_types.get(c) in ('numeric', 'categorical')]
                d.interaction_df = frame
                d._remap_ids()
                values = d.interaction_df[outputs].apply(pd.to_numeric, errors='raise').fillna(0).to_numpy(dtype=np.float32)
                for c in outputs:
                    if c not in union:
                        union.append(c)
                prepared.append((path, outputs, values, steps))
            self.union_columns[feature_index] = union
            self.widths.append(len(union))
            for path, outputs, values, steps in prepared:
                # COO assembly avoids a dense [rows, union_width] staging matrix.
                rows, cols = np.nonzero(values)
                mapped = np.asarray([union.index(c) for c in outputs], dtype=np.int64)
                self.blocks[feature_index, path] = sparse.csr_matrix(
                    (values[rows, cols], (rows, mapped[cols])), shape=(len(self.frame), len(union)))
                self.columns[feature_index, path] = outputs
                self.step_params[feature_index, path] = steps
        self.output_dim = sum(self.widths)

    def forward(self, row_indices, beta, alpha=None):
        indices = row_indices[:, 0].detach().cpu().numpy().astype(np.int64)
        groups = []
        for feature_index in range(len(self.feature_names)):
            mixed = None
            total_probability = beta.new_zeros(())
            for path in self.feature_paths[feature_index]:
                probability = beta.new_ones(())
                order, choices = path
                for slot_index, candidate in zip(self.learned_slots, choices):
                    probability = probability * beta[slot_index, feature_index, candidate]
                if alpha is not None:
                    for position, slot_index in enumerate(order):
                        probability = probability * alpha[feature_index, position, slot_index]
                block = torch.as_tensor(self.blocks[feature_index, path][indices].toarray(),
                                        dtype=beta.dtype, device=beta.device)
                total_probability = total_probability + probability
                weighted = probability * block
                mixed = weighted if mixed is None else mixed + weighted
            groups.append(mixed / total_probability.clamp(min=1e-12))
        return torch.cat(groups, dim=1)

    def selected_steps(self, continuous):
        """Use the exact parameters of the transformations optimized in search."""
        beta = continuous.beta().detach().cpu()
        result = []
        # Preserve stage order. Independent columns in the same stage commute.
        from .discretizer import slot_order
        for slot_index in slot_order(continuous):
            if slot_index not in self.learned_slots:
                continue
            for f in range(len(self.feature_names)):
                candidate = int(beta[slot_index, f, :self.slots[slot_index].n_candidates].argmax())
                op_name = self.slots[slot_index].candidates[candidate]
                if op_name == IDENTITY:
                    continue
                # Every path uses identical seeded params for a stage/feature.
                step = diffprep_make_step(op_name, continuous.rec_context, random.Random(continuous.rec_seed))
                import inspect
                if 'cols' in inspect.signature(continuous.rec_data._load_op(op_name).__init__).parameters:
                    step.params['cols'] = [self.feature_names[f]]
                elif 'boundaries' in step.params:
                    step.params['boundaries'] = {self.feature_names[f]: next(iter(step.params['boundaries'].values()))}
                result.append(step)
        return result

    def contract(self):
        return {'adaptation': 'exact_real_transform_path_mixture_v1',
                'feature_names': self.feature_names, 'padded_output_dim': self.output_dim,
                'paths_per_feature': {self.feature_names[f]: len(v) for f, v in self.feature_paths.items()},
                'column_eligibility': 'numeric transforms on numeric; category encoders on categorical; no operator family removed',
                'training_identity': hashlib.sha256(pd.util.hash_pandas_object(
                    self.frame.loc[self.frame['__split__'].eq('train'), ['user_id', 'item_id', 'timestamp']].astype('string'),
                    index=False).to_numpy().tobytes()).hexdigest()}
