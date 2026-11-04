"""Dataset-declared clocks and static applicability; no value/column-name guesses."""
from copy import deepcopy

TIME_OPS = {'ParseDate', 'ExtractDateTimeFeature'}
HISTORY_OPS = {'CreateLagFeature', 'CreateRollingFeature'}


def declared_history(dataset, params, history=None):
    if history:
        return deepcopy(history)
    # Existing rec_train_only_v1 semantics, declared for these datasets only.
    # Keep its global default and explicitly supported user/item groupings.
    names = {'amazon_beauty', 'AmazonBeauty', 'movielens', 'Movielens', 'MovieLens',
             'yelp', 'Yelp', 'kuairec', 'Kuairec', 'KuaiRec'}
    if dataset in names:
        groups = params.get('group_cols') or []
        if groups in ([], ['user_id'], ['item_id'], ['user_id', 'item_id']):
            return dict(entity_cols=groups, time_col='timestamp', ordering='ascending',
                        same_time='exclude', mode='training_only', availability='event_time')
    return {}


def declared_clock(dataset, column, rules=None):
    from .tabular_contracts import time_rule
    from .rec_time_context import source_time_rule
    if dataset in ('elliptic_bitcoin', 'EllipticBitcoin') and column == 'time_step':
        return {'kind': 'ordered_step'}
    explicit = (rules or {}).get(column)
    rule = explicit or time_rule(dataset, column) or source_time_rule(dataset, column)
    if not rule and dataset in ('berka', 'Berka') and column == 'birth_number':
        rule = {'kind': 'numeric_date', 'mode': 'berka_birth'}
    if not rule:
        return None
    rule = deepcopy(rule)
    # Existing format/unit contracts are explicit declarations, not dtype guesses.
    rule.setdefault('kind', 'numeric_date' if rule.get('format', '').startswith('%y%m%d')
                    or rule.get('format', '').startswith('%Y%m%d') else 'calendar')
    return rule


def resolve_time_params(op, params, dataset, rules=None, history=None, target=None):
    """Return effective parameters and static errors shared by all entry points.

    Malformed values, missing actual data, and execution/resource failures remain
    runtime checks. Operator rules may specify parsing but cannot override an
    ordered-step declaration or invent dataset semantics.
    """
    p = deepcopy(params)
    errors = []
    if op in TIME_OPS:
        cols = p.get('cols') or []
        cols = [cols] if isinstance(cols, str) else cols
        for col in cols:
            clock = declared_clock(dataset, col, rules)
            if not clock or clock.get('kind') not in ('calendar', 'numeric_date'):
                errors.append(f'{op}: {col!r} has no declared calendar/date semantics')
                continue
            override = (p.get('time_rules') or {}).get(col)
            for key in ('kind', 'format', 'unit', 'origin', 'source_date', 'source_hour'):
                if key in (override or {}) and key in clock and override[key] != clock[key]:
                    errors.append(f'{op}: {col!r} {key} conflicts with dataset clock')
            rule = dict(clock, **(override or {}))
            mode = p.get('mode', 'string') if op == 'ParseDate' else 'string'
            if mode in ('int_yymmdd', 'berka_birth'):
                if clock.get('mode') != mode and not (mode == 'int_yymmdd' and clock.get('format') == '%y%m%d'):
                    errors.append(f'{op}: mode {mode!r} conflicts with declared clock for {col!r}')
            elif mode != 'string':
                errors.append(f'{op}: unsupported mode {mode!r}')
            elif 'unit' in rule:
                if 'origin' not in rule or rule.get('format'):
                    errors.append(f'{op}: epoch parsing requires unit/origin without format')
            elif rule.get('representation') == 'datetime':
                rule['format'] = 'mixed'
            elif not rule.get('format'):
                errors.append(f'{op}: calendar parsing requires an explicit format')
            p.setdefault('time_rules', {})[col] = {k: v for k, v in rule.items() if k not in ('kind', 'mode')}
    elif op in HISTORY_OPS:
        contract = declared_history(dataset, p, history)
        contract.update(contract.get('sources', {}).get(p.get('target_col'), {}))
        if not contract:
            return p, [f'{op}: dataset history protocol is undeclared']
        required = {'ordering': 'ascending', 'same_time': 'exclude', 'mode': 'training_only'}
        for key, value in required.items():
            if contract.get(key) != value:
                errors.append(f'{op}: history requires declared {key}={value}')
        for parameter, field in [('group_cols', 'entity_cols'), ('time_col', 'time_col'),
                                 ('availability', 'availability'), ('available_at_col', 'available_at_col')]:
            declared = contract.get(field)
            if p.get(parameter) is not None and declared != p[parameter]:
                errors.append(f'{op}: {parameter} conflicts with dataset history protocol')
            if declared is not None:
                p[parameter] = declared
        if not isinstance(p.get('group_cols'), list) or not p.get('time_col'):
            errors.append(f'{op}: explicit entity_cols (including [] for single series) and time_col required')
        effective_rules = dict(rules or {})
        if contract.get('time_rule') and p.get('time_col'):
            effective_rules[p['time_col']] = contract['time_rule']
        clock = declared_clock(dataset, p.get('time_col'), effective_rules)
        if not clock or clock.get('kind') not in ('calendar', 'numeric_date', 'ordered_step'):
            errors.append(f'{op}: history clock is undeclared')
        if p.get('target_col') == target:
            errors.append(f'{op}: supervised label history is not supported by this feature protocol')
        if not p.get('available_at_col') and p.get('availability') != 'event_time':
            errors.append(f'{op}: source availability must be explicitly declared')
        if p.get('available_at_col') and not declared_clock(dataset, p['available_at_col'], rules):
            errors.append(f'{op}: availability clock is undeclared')
        if p.get('available_at_col'):
            event_clock = clock or {}
            available_clock = declared_clock(dataset, p['available_at_col'], rules) or {}
            if (event_clock.get('kind') == 'ordered_step') != (available_clock.get('kind') == 'ordered_step'):
                errors.append(f'{op}: event and availability clocks are not comparable')
    return p, errors
