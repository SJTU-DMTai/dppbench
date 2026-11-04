"""Conservative, semantics-preserving compaction of independent imputations."""
import copy


def compact_column_steps(pipeline):
    """Merge adjacent simple HandleMV nodes, never forks or coupled imputers.

    Simple imputation fits and transforms each explicit column independently.
    Matching parameters and disjoint columns therefore preserve fitted values,
    column dtypes and train/held-out behavior. KNN/iterative are not separable.
    """
    def key(step):
        p = step.params
        cols = p.get('cols')
        if (step.op != 'HandleMV' or not isinstance(cols, list) or not cols
                or not all(isinstance(c, str) for c in cols)
                or len(set(cols)) != len(cols)):
            return None
        allowed = {'cols', 'method', 'action', 'n_neighbors', 'max_iter', 'fill_value', 'na_values'}
        if set(p) - allowed:
            return None
        defaults = dict(method='median', action='impute', n_neighbors=5,
                        max_iter=10, fill_value=None, na_values=[])
        defaults.update({k: v for k, v in p.items() if k != 'cols'})
        if defaults['action'] != 'impute' or defaults['method'] not in ('mean', 'median', 'mode', 'constant'):
            return None
        return step.target, defaults

    consumers = {}
    ids = [step.id for step in pipeline.steps if step.id is not None]
    if len(ids) != len(set(ids)):
        return pipeline  # Leave malformed DAGs intact for the validator.
    for step in pipeline.steps:
        for parent in step.prev:
            consumers[parent] = consumers.get(parent, 0) + 1
    for parent in pipeline.train_prev or []:
        consumers[parent] = consumers.get(parent, 0) + 1
    result = []
    for original in pipeline.steps:
        step = original
        left = result[-1] if result else None
        plain = left is not None and not left.id and not step.id and not left.prev and not step.prev
        chain = (left is not None and left.id and step.id and step.prev == [left.id]
                 and consumers.get(left.id) == 1)
        if (left is not None and (plain or chain) and key(left) is not None
                and key(left) == key(step)
                and set(left.params['cols']).isdisjoint(step.params['cols'])):
            step = copy.deepcopy(step)
            step.params['cols'] = left.params['cols'] + step.params['cols']
            step.prev = left.prev
            result[-1] = step
        else:
            result.append(step)
    pipeline.steps = result
    return pipeline
