import copy
import random

from .pipeline import Pipeline, make_step, _strip_df_params
from .utils import get_available_ops
from .op_spec import is_mandatory


VERSION = 'shared-repair-v1'


def complete_pipeline(pipeline, ctx, seed=42, max_len=10):
    """Add missing basics without removing or changing existing operator instances."""
    if not pipeline.steps:
        raise ValueError('Shared repair requires a nonempty parsed candidate')
    result = Pipeline.from_yaml(pipeline.to_yaml())
    allowed = {spec.op_name for spec in get_available_ops(ctx.task_type)}
    if any(step.op not in allowed for step in result.steps):
        raise ValueError('Shared repair cannot replace unsupported operators')
    required = {spec.op_name for spec in get_available_ops(ctx.task_type)
                if is_mandatory(spec.op_name, ctx.task_type)}
    if ctx.secondary_tables:
        required.add('JoinTable')
    nodes = {step.id: step for step in result.steps}
    reachable = set()
    pending = list(result.train_prev)
    while pending:
        ref = pending.pop()
        if ref in reachable:
            continue
        reachable.add(ref)
        if ref in nodes:
            pending.extend(nodes[ref].prev)
    present = {node.op for ref, node in nodes.items() if ref in reachable}
    missing = required - present
    if len(result.steps) + len(missing) > max_len:
        raise ValueError('Shared repair would exceed max_pipeline_len')
    if len(result.train_prev) != 1:
        raise ValueError('Shared repair requires a single primary output')
    sources = {source['table']: source['id'] for source in result.sources}
    if ctx.primary_table not in sources:
        raise ValueError('Shared repair requires the primary source')
    used = set(nodes) | set(sources.values())
    rng = random.Random(seed)
    added = []

    def allocate(prefix):
        index = 1
        while f'{prefix}{index}' in used:
            index += 1
        name = f'{prefix}{index}'
        used.add(name)
        return name

    for name in ['JoinTable', 'HandleMV', 'LabelEncode', 'CreateSequence'] + sorted(missing - {'JoinTable', 'HandleMV', 'LabelEncode', 'CreateSequence'}):
        if name not in missing:
            continue
        step = make_step(name, ctx, rng)
        if step is None:
            raise ValueError(f'Cannot instantiate required operator {name}')
        params, refs = _strip_df_params(step.params)
        step.params = params
        step.id = allocate('repair_op')
        if name == 'JoinTable':
            primary = sources[ctx.primary_table]
            step.prev = [primary]
            for table in refs:
                if table not in sources:
                    # Relation DAG execution requires source IDs matching s<number>.
                    source = {'id': allocate('s'), 'table': table}
                    result.sources.append(source)
                    sources[table] = source['id']
                step.prev.append(sources[table])
            if len(step.prev) < 2:
                raise ValueError('Cannot bind a missing JoinTable to a secondary source')
            for original in result.steps:
                if original.prev and original.prev[0] == primary:
                    original.prev[0] = step.id
            result.steps.insert(0, step)
            result.train_prev = [step.id if ref == primary else ref for ref in result.train_prev]
        else:
            step.prev = list(result.train_prev)
            result.steps.append(step)
            result.train_prev = [step.id]
        added.append(step.to_dict())
    report = {'version': VERSION, 'raw_hash': pipeline.hash(), 'effective_hash': result.hash(),
              'added': copy.deepcopy(added)}
    return result, report
