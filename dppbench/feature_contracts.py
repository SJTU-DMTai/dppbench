"""Shared supervised-feature selectors; defaults and explicit YAML obey one guard."""

FEATURE_SELECTORS = {
    "FrequencyEncode": ("cols",), "HashEncode": ("cols",),
    "OneHotEncode": ("cols",), "OrdinalEncode": ("cols",),
    "LabelEncode": ("cols",), "TargetEncode": ("cols",),
    "DiscretizeFeature": ("cols", "boundaries"),
    "CreateFeature": ("source_cols",), "CrossFeature": ("cols",),
    "CreatePolynomialFeature": ("cols",), "ReduceDimension": ("cols",),
    "AggregateGroupFeature": ("group_cols", "agg_cols"),
    "CreateLagFeature": ("target_col", "group_cols"),
    "CreateRollingFeature": ("target_col", "group_cols"),
    "ExtractDateTimeFeature": ("cols",), "ExtractTextFeature": ("cols",),
    "ExtractTextEmbedding": ("cols",), "ExtractGraphFeature": ("source_col", "target_col"),
    "SplitColumn": ("col",), "RenameColumn": ("rename_map",),
    "CastType": ("col_dtypes",), "ParseNumber": ("cols",), "ParseDate": ("cols",),
    "ScaleFeature": ("cols",), "TransformPower": ("cols",),
    "ClipOutlier": ("cols",), "HandleOutlier": ("cols",),
}


def _names(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_names(v) for pair in value.items() for v in pair)) if value else set()
    if isinstance(value, (list, tuple, set)):
        return set().union(*(_names(v) for v in value)) if value else set()
    return set()


def target_feature_errors(op_name, params, target):
    if not target:
        return []
    keys = FEATURE_SELECTORS.get(op_name, ())
    bad = [k for k in keys if target in _names(params.get(k))]
    diagnostic = {'sample_weight', '__split__', f'{target}_corrected'}
    bad += [k for k in keys if diagnostic.intersection(_names(params.get(k)))]
    # A new feature must not overwrite the label either.
    outputs = ("output_col", "output_cols", "seq_col")
    bad += [k for k in outputs if ({target} | diagnostic).intersection(_names(params.get(k)))]
    cols = params.get('cols') or []
    cols = [cols] if isinstance(cols, str) else cols
    generated = []
    if op_name == 'FrequencyEncode':
        generated = [f"{c}{params.get('suffix', '_freq')}" for c in cols]
    elif op_name == 'HashEncode':
        generated = [f"{params.get('prefix', 'h_')}{c}" for c in cols]
    elif op_name == 'TargetEncode':
        generated = [f"{c}_{params.get('target_col', 'rating')}" for c in cols]
    if ({target} | diagnostic).intersection(generated):
        bad.append('generated output name')
    return ([f"{op_name} cannot use supervised target {target!r} as a feature ({', '.join(bad)})"]
            if bad else [])
