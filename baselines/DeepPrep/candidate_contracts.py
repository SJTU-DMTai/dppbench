"""LLM proposal validation, without changing shared operator semantics."""
from baselines.common.pipeline import (
    assign_dag_structure as _assign, check_pipeline as _check,
    PipelineCheckReport,
)


def parameter_errors(op, params, ctx=None):
    errors = []
    if op == 'JoinTable':
        funcs = params.get('agg_funcs')
        allowed = {'mean', 'max', 'min', 'sum', 'std', 'var', 'median', 'count', 'nunique', 'first', 'last'}
        if funcs is not None and (not isinstance(funcs, list) or not funcs or
                any(not isinstance(f, str) or f not in allowed for f in funcs)):
            errors.append('JoinTable agg_funcs must be a nonempty list such as ["mean","min","max"], not a nested dictionary')
    if ctx is None:
        return errors
    protected = {c for c in (ctx.target_col, ctx.id_col, ctx.time_col) if c}
    if op == 'DropColumns':
        cols = params.get('cols') or []
        cols = [cols] if isinstance(cols, str) else cols
        if protected.intersection(cols):
            errors.append(f'DropColumns cannot drop protected columns: {sorted(protected.intersection(cols))}')
    if op == 'RenameColumn':
        mapping = params.get('rename_map', {})
        if isinstance(mapping, dict) and any(c in protected and mapping[c] != c for c in mapping):
            errors.append('RenameColumn cannot rename target, identifier, or time columns')
    if op == 'JoinTable' and params.get('method', 'key') in {'key', 'agg'}:
        ref = params.get('aux_df')
        if isinstance(ref, str):
            name = ref.lstrip('$')
            if name not in ctx.tables and name.endswith('_df'):
                name = name[:-3]
            table = ctx.tables.get(name)
            key = params.get('key_col')
            if table is None or name == ctx.primary_table:
                errors.append('JoinTable aux_df must reference an available independent auxiliary table')
            elif key not in table.columns:
                errors.append(f'JoinTable right key {key!r} is absent from {name!r}')
            elif params.get('method', 'key') == 'key' and table.join_key_unique.get(key) is False:
                errors.append(f'JoinTable right key {key!r} is repeated; use method="agg" with a distinct prefix')
    return errors


def assign_dag_structure(pipeline, ctx, rng=None):
    # A deterministic parameter chosen by the LLM adapter, not a new operator.
    for step in pipeline.steps:
        if step.op == 'JoinTable' and step.params.get('method') == 'agg' and not step.params.get('prefix'):
            ref = step.params.get('aux_df')
            if isinstance(ref, str):
                name = ref.lstrip('$')
                if name not in ctx.tables and name.endswith('_df'):
                    name = name[:-3]
                step.params['prefix'] = name + '__'
    return _assign(pipeline, ctx, rng)


def check_pipeline(pipeline, ctx=None, **kwargs):
    report = _check(pipeline, ctx, **kwargs)
    errors = list(report.errors)
    if hasattr(pipeline, 'steps'):
        for step in pipeline.steps:
            errors.extend(parameter_errors(step.op, step.params or {}, ctx))
    return PipelineCheckReport(not errors, errors, report.warnings)


def frame_errors(steps, ctx, frame):
    """Check column proposals against current sandbox state before executing."""
    import pandas as pd
    if not isinstance(frame, pd.DataFrame):
        return []
    if not frame.columns.is_unique:
        return ['Candidate input has duplicate column names']
    columns = set(frame.columns)
    exact = True
    errors = []
    stable = {'HandleMV', 'LabelEncode', 'SortRows', 'CastType', 'ScaleFeature',
              'ClipOutlier', 'HandleOutlier', 'Deduplicate', 'FilterSample'}
    for step in steps:
        p = step.params or {}
        errors.extend(parameter_errors(step.op, p, ctx))
        if step.op == 'RenameColumn' and isinstance(p.get('rename_map'), dict):
            mapping = p['rename_map']
            if exact and set(mapping) - columns:
                errors.append(f'RenameColumn source columns do not exist: {sorted(set(mapping) - columns)}')
            renamed = [mapping.get(c, c) for c in columns]
            if len(renamed) != len(set(renamed)) or len(set(mapping.values())) != len(mapping):
                errors.append('RenameColumn would create duplicate column names')
            columns = set(renamed)
        elif step.op == 'DropColumns':
            cols = p.get('cols') or []
            columns.difference_update([cols] if isinstance(cols, str) else cols)
        elif step.op == 'JoinTable' and p.get('method', 'key') in {'key', 'agg'}:
            if exact and p.get('key_col') not in columns:
                errors.append(f'JoinTable missing join key {p.get("key_col")!r} in main table')
            exact = False
        elif step.op not in stable:
            exact = False
    return errors
