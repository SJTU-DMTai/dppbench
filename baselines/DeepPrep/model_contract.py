"""Expose the actual model input contract without changing candidate semantics."""
import yaml


def sequence_input_errors(pipeline, ctx, config):
    if ctx.task_type != 'rec' or config.get('model') not in {'DIN', 'DIEN', 'SIM'}:
        return []
    mappings = config.get('feature', {}).get('seq_cols', {}) or {}
    dag = yaml.safe_load(pipeline.to_yaml())['dag']
    # Track available scalar/list roles per branch. Do not invent bindings for
    # arbitrary custom sequence names, or change the model configuration.
    schemas = {s['id']: {c: t.semantic_types.get(c, 'scalar')
                         for c in t.columns}
               for s in dag['sources'] if (t := ctx.tables.get(s['table'])) is not None}
    for node in dag['ops']:
        schema = {}
        for parent in node['prev']:
            schema.update(schemas.get(parent, {}))
        p = node.get('params') or {}
        if node['op'] == 'CreateSequence':
            schema[p.get('seq_col', 'item_id_seq')] = 'categorical_list'
        elif node['op'] == 'DropColumns':
            cols = p.get('cols') or []
            for c in ([cols] if isinstance(cols, str) else cols):
                schema.pop(c, None)
        elif node['op'] == 'RenameColumn':
            for old, new in (p.get('rename_map') or {}).items():
                if old in schema:
                    schema[new] = schema.pop(old)
        schemas[node['id']] = schema
    schema = schemas.get(dag['train']['prev'][0], {})
    present = {seq: ref for seq, ref in mappings.items() if schema.get(seq) == 'categorical_list'}
    bad = {seq: ref for seq, ref in present.items()
           if ref not in schema or schema[ref] == 'categorical_list'}
    if bad:
        return [f"{config['model']} input contract: attention sequences have missing or non-scalar query features: {bad}. Preserve each present sequence's configured query feature."]
    if present:
        return []
    produced = [c for c, kind in schema.items() if kind == 'categorical_list']
    return [f"{config['model']} input contract: no configured attention sequence/query pair is available. "
            f"Configured sequence mappings: {mappings}; available list columns: {produced}. "
            "Create a configured sequence with its referenced scalar feature preserved; arbitrary sequence names are not automatically bound. Empty per-row histories are permitted."]
