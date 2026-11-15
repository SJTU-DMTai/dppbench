"""Read-only adapter checks; never repair a model's proposed strategy."""
from baselines.common.pipeline import PipelineCheckReport, check_pipeline


def check_candidate(pipeline, ctx):
    report = check_pipeline(pipeline, ctx)
    errors = list(report.errors)
    # Both TabularData and RecData execute relation DAGs with one output.
    if pipeline.train_prev is not None and len(pipeline.train_prev) != 1:
        errors.append('dag.train.prev must contain exactly one node; connect or remove unused branches, never list multiple training outputs')
    for step in pipeline.steps:
        cols = step.params.get('cols')
        cols = [cols] if isinstance(cols, str) else (cols or [])
        if step.op == 'HandleMV' and ctx.target_col in cols:
            errors.append(f'HandleMV {step.id}: cannot consume supervised target {ctx.target_col!r}; choose feature columns explicitly')
    return PipelineCheckReport(not errors, errors, report.warnings)


def task_contract(ctx, model_config=None):
    from baselines.DeepPrep.operator_catalog import mandatory_ops_for_context
    return {
        'target_col': ctx.target_col, 'user_col': ctx.user_col if ctx.task_type == 'rec' else None,
        'item_col': ctx.item_col if ctx.task_type == 'rec' else None,
        'time_col': ctx.time_col, 'id_col': ctx.id_col,
        'mandatory_ops': mandatory_ops_for_context(ctx),
        'model': (model_config or {}).get('model'),
        'sequence_mappings': (model_config or {}).get('feature', {}).get('seq_cols', {}),
        'rules': [
            'The supervised label is already prepared by the loader. Do not parse, impute, encode, scale or create history features from it.',
            'Exactly one train.prev output; every retained operator must reach it. Delete unused branches or connect them through supported operators.',
            'Preserve ID, time, split and graph alignment. No automatic candidate repair is performed.',
            'Correct every reported error in the full strategy; appending operators does not fix an invalid prefix.',
        ],
    }
