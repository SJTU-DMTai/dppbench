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

    @classmethod
    def from_context(cls, ctx: DataContext) -> "TargetSchema":
        if ctx.task_type == "rec":
            required = tuple(c for c in (ctx.user_col, ctx.item_col, ctx.target_col) if c)
            tables = tuple(ctx.source_tables())
            constraints = (
                "preserve interaction row alignment",
                "preserve user, item, and label identifiers",
                "produce model-consumable feature dtypes",
            )
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
        return "\n".join(lines)

    def validate_observation(self, obs) -> list[str]:
        errors: list[str] = []
        tables = getattr(obs, "tables", {}) or {}
        available_tables = set(tables)
        for name in self.required_tables:
            if name not in available_tables:
                errors.append(f"required table {name!r} is missing")
        all_columns = {
            str(col)
            for table in tables.values()
            for col in table.get("columns", [])
        }
        for col in self.required_columns:
            if col not in all_columns:
                errors.append(f"required column {col!r} is missing")
        return errors


__all__ = ["TargetSchema"]
