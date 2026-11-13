"""DeepPrep-only recommendation lookup validation; shared contracts stay intact."""
import yaml
from baselines.common.pipeline import Pipeline, PipelineCheckReport, check_pipeline


def lookup_reference_errors(op_name, params, ctx):
    if ctx is None or ctx.task_type != "rec" or op_name != "JoinTable":
        return []
    errors = []
    for key in ("aux_df", "user_df", "item_df"):
        value = (params or {}).get(key)
        if not isinstance(value, str) or not value:
            continue
        table = value.lstrip("$")
        table = table[:-3] if table.endswith("_df") else table
        if table == ctx.primary_table:
            errors.append(f"JoinTable: {key} cannot reference primary interaction table {value}; use an independent side table")
        elif table not in ctx.tables:
            errors.append(f"JoinTable: {key} references unavailable side table {value}")
        elif (params or {}).get("method") == "rec" and key in ("user_df", "item_df") and table != key[:-3]:
            errors.append(f"JoinTable: {key} must reference ${key}, not {value}")
    return errors


def check_deepprep_pipeline(pipeline, ctx=None, **kwargs):
    report = check_pipeline(pipeline, ctx, **kwargs)
    if ctx is None or ctx.task_type != "rec":
        return report
    errors = list(report.errors)
    if isinstance(pipeline, Pipeline):
        for step in pipeline.steps:
            errors.extend(lookup_reference_errors(step.op, step.params, ctx))
        data = yaml.safe_load(pipeline.to_yaml())
    elif isinstance(pipeline, str):
        try:
            data = yaml.safe_load(pipeline)
        except yaml.YAMLError:
            return report
    else:
        data = pipeline
    dag = data.get("dag", {}) if isinstance(data, dict) else {}
    if not isinstance(dag, dict):
        return report
    lineage = {source.get("id"): source.get("table") == ctx.primary_table
               for source in dag.get("sources", []) if isinstance(source, dict)}
    for node in dag.get("ops", []):
        if not isinstance(node, dict):
            continue
        prev = node.get("prev", [])
        if not isinstance(prev, list) or not all(isinstance(pid, str) for pid in prev):
            continue
        if node.get("op") == "JoinTable":
            errors.extend(lookup_reference_errors("JoinTable", node.get("params"), ctx))
            if prev and lineage.get(prev[0], False) and any(lineage.get(pid, False) for pid in prev[1:]):
                errors.append(f"JoinTable {node.get('id')}: partitioned primary branches cannot be secondary inputs; explicit split-aware alignment is required")
        lineage[node.get("id")] = any(lineage.get(pid, False) for pid in prev)
    return PipelineCheckReport(not errors, errors, list(report.warnings))
