"""DPPBench target-schema adapter for DeepPrep inference.

DeepPrep expects a target schema Sigma*. DPPBench does not ship that object,
so it is derived only from the public task/model contract. The task's
``prepare.yaml`` is deliberately not read or exposed to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from baselines.common.data_ctx import DataContext


@dataclass(frozen=True)
class TargetSchema:
    task_type: str
    primary_table: str
    target_col: str | None
    required_columns: tuple[str, ...] = ()
    required_tables: tuple[str, ...] = ()
    constraints: tuple[str, ...] = field(default_factory=tuple)
    attention_sequences: dict[str, str] = field(default_factory=dict)
    seq_max_len: int = 20

    @classmethod
    def from_context(cls, ctx: DataContext, model_config: dict | None = None) -> "TargetSchema":
        config = model_config or {}
        feature = config.get('feature', {})
        attention_sequences = (dict(feature.get('seq_cols', {}))
                               if str(config.get('model', '')).upper() in ('DIN', 'DIEN', 'SIM') else {})
        if ctx.task_type == "rec":
            required = tuple(c for c in (ctx.user_col, ctx.item_col, ctx.target_col) if c)
            tables = tuple(ctx.source_tables())
            constraints = (
                "preserve interaction row alignment",
                "preserve user, item, and label identifiers",
                "produce model-consumable feature dtypes",
            )
        elif ctx.task_type == 'graph':
            required = tuple(c for c in (ctx.id_col, ctx.time_col, ctx.target_col) if c)
            tables = (ctx.primary_table,)
            constraints = ('preserve node IDs, event times, labels, and node alignment',
                           'keep unknown-label nodes; exclude them from supervised loss',
                           'preserve topology and frozen-test isolation; do not resample graph nodes',
                           'satisfy mandatory DPPBench operators')
        else:
            required = tuple(c for c in (ctx.target_col,) if c)
            tables = (ctx.primary_table,)
            constraints = (
                "preserve the target column and its row alignment",
                "produce model-consumable feature dtypes",
                "satisfy all mandatory DPPBench operators",
            )
        return cls(
            task_type=ctx.task_type,
            primary_table=ctx.primary_table,
            target_col=ctx.target_col,
            required_columns=required,
            required_tables=tables,
            constraints=constraints,
            attention_sequences=attention_sequences,
            seq_max_len=int(feature.get('seq_max_len', 20)),
        )

    def render(self) -> str:
        lines = [
            f"task_type: {self.task_type}",
            f"primary_table: {self.primary_table}",
            f"target_col: {self.target_col}",
            f"required_tables: {list(self.required_tables)}",
            f"required_columns: {list(self.required_columns)}",
            "constraints:",
        ]
        lines.extend(f"  - {item}" for item in self.constraints)
        if self.attention_sequences:
            lines.append(f"attention_sequences (sequence column -> categorical query column): {self.attention_sequences}")
            lines.append(f"seq_max_len: {self.seq_max_len}")
            lines.append("The primary table must contain at least one valid attention sequence and its query column. "
                         "Use these exact sequence names; an alias such as item_history does not satisfy this contract.")
        return "\n".join(lines)

    def validate_observation(self, obs) -> list[str]:
        errors: list[str] = []
        tables = getattr(obs, "tables", {}) or {}
        available_tables = set(tables)
        for name in self.required_tables:
            if name not in available_tables:
                errors.append(f"required table {name!r} is missing")
        columns = set(getattr(obs, 'columns', []))
        for col in self.required_columns:
            if col not in columns:
                errors.append(f"required column {col!r} is missing from primary table")
        if self.attention_sequences:
            col_types = getattr(obs, 'col_types', {})
            valid = []
            for sequence, query in self.attention_sequences.items():
                if sequence not in columns:
                    continue
                if col_types.get(sequence) != 'categorical_list' or col_types.get(query) != 'categorical' or query not in columns:
                    errors.append(f"attention sequence {sequence!r} must be categorical_list and reference "
                                  f"categorical query column {query!r} in the primary table")
                else:
                    valid.append(sequence)
            if not valid:
                errors.append(f"No valid attention sequence in primary table. Required mapping: {self.attention_sequences}. "
                              "Revise CreateSequence using an exact configured seq_col and item_col; do not use an alias.")
        return errors


__all__ = ["TargetSchema"]
