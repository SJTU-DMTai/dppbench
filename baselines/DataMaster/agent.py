"""Main DataMaster agent loop.

Implements the high-level pseudo-code from the paper §3.5:

    while budget_left and tree.has_frontier():
        v = scheduler.select(tree, step)
        for k in range(K_black):
            <solution> = LLM.propose(memory.retrieve(v), op_catalog, parent.acc)
            child_steps = parse_chain(<solution>)
            child_pipeline = repair(parent.acc + child_steps)
            (y, phi) = evaluator.evaluate_for_agent(child_pipeline)
            tree.add_black_child(v, child_steps, y, phi)
            backpropagate(tree, child, scheduler.compute_reward(child, v))

The black-only growth policy and the LLM-as-operator-selector adaptation
follow the project requirements (no Red nodes, no codegen).
"""
from __future__ import annotations

import copy
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from baselines.common.pipeline import DataContext, Pipeline
from baselines.common.pipeline_constraints import is_legal, repair
from baselines.DeepPrep.tree_agent import (
    ChainParseError,
    chain_to_steps,
    parse_response,
)
from baselines.DeepPrep.sandbox import Sandbox
from baselines.DeepPrep.llm_client import LLMClient

from .data_tree import DataTree, backpropagate
from .memory import GlobalMemory
from .operator_catalog import format_op_descriptions
from .prompts import (
    SYSTEM_DATAMASTER,
    render_retry_feedback,
    render_user_initial,
)
from .scheduler import UCBScheduler


logger = logging.getLogger(__name__)


_FINDING_RE = re.compile(r"<finding>(.*?)</finding>", re.DOTALL)


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------
@dataclass
class DataMasterRunResult:
    pipeline: Pipeline
    success: bool
    n_iterations: int
    n_expansions: int
    n_errors: int
    best_node_id: Optional[str]
    transcript: list[dict] = field(default_factory=list)
    solution_attempts: list[dict] = field(default_factory=list)
    tree: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DataMasterAgent:
    """Black-node-only DataMaster search.

    Parameters
    ----------
    llm
        DeepPrep's :class:`LLMClient` (api or local backend).
    sandbox
        DeepPrep's :class:`Sandbox`. Used to (a) re-play the parent's
        accumulated steps before each expansion and (b) sanity-check that
        the candidate chain is structurally executable.
    ctx
        SAGA :class:`DataContext` (numeric/categorical/list/text columns,
        target_col, time_col, ...). Drives ``chain_to_steps`` defaults.
    tree, memory, scheduler
        DataMaster's three core components (paper §3.2).
    downstream_evaluator
        Callable that scores a Pipeline -> ``(fitness, metrics, error)``.
        DataMaster requires this; without a downstream metric the UCB
        backpropagation cannot work.
    """

    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        ctx: DataContext,
        tree: DataTree,
        memory: GlobalMemory,
        scheduler: UCBScheduler,
        downstream_evaluator: Callable[
            [Pipeline], tuple[Optional[float], dict, Optional[str]]
        ],
        *,
        max_iterations: int = 5,
        k_black: int = 3,
        max_chain_len: int = 6,
        max_depth: int = 6,
        max_err_cnt: int = 6,
        max_solution_attempts: int = 2,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.sandbox = sandbox
        self.ctx = ctx
        self.tree = tree
        self.memory = memory
        self.scheduler = scheduler
        self.downstream_evaluator = downstream_evaluator

        self.max_iterations = int(max_iterations)
        self.k_black = max(1, int(k_black))
        self.max_chain_len = int(max_chain_len)
        self.max_depth = int(max_depth)
        self.max_err_cnt = int(max_err_cnt)
        self.max_solution_attempts = max(1, int(max_solution_attempts))
        self.verbose = bool(verbose)

        self._rng = random.Random(seed)
        self.n_errors = 0
        self.transcript: list[dict] = []
        self.solution_attempts: list[dict] = []

    # ------------------------------------------------------------------
    def _log(self, *args, **kwargs) -> None:
        if self.verbose:
            print("[DataMaster.Agent]", *args, **kwargs)

    # ------------------------------------------------------------------
    def _summary_text(self) -> str:
        ctx = self.ctx
        return (
            f"task_type={ctx.task_type} dataset={ctx.data_name}\n"
            f"target_col={ctx.target_col} time_col={ctx.time_col} id_col={ctx.id_col}\n"
            f"#numeric_cols={len(ctx.numeric_cols)} "
            f"#categorical_cols={len(ctx.categorical_cols)} "
            f"#list_cols={len(ctx.list_cols)} #text_cols={len(ctx.text_cols)}\n"
            f"has_user_df={ctx.has_user_df} has_item_df={ctx.has_item_df} "
            f"aux_dfs={ctx.aux_dfs}"
        )

    # ------------------------------------------------------------------
    def _replay_parent(self, parent_acc_steps) -> tuple[bool, str, Optional[str]]:
        """Re-load the sandbox to root then replay the parent pipeline.

        Returns (ok, obs_text, error). On replay failure we still continue
        with whatever observation we managed to render so the LLM can react.
        """
        try:
            reset = self.sandbox.reset()
        except Exception as e:  # pragma: no cover - sandbox normally robust
            return False, f"(sandbox reset failed: {e})", str(e)
        if not reset.success or reset.obs is None:
            return False, "(sandbox reset failed)", reset.error or "unknown"
        if not parent_acc_steps:
            return True, reset.obs.text, None
        exec_res = self.sandbox.execute_chain(list(parent_acc_steps))
        if not exec_res.success or exec_res.obs is None:
            return False, reset.obs.text, exec_res.error
        return True, exec_res.obs.text, None

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        *,
        parent_obs_text: str,
        parent_ops: list[str],
        parent_fitness: Optional[float],
        memory_context: str,
        expansion_idx: int,
        step: int,
    ) -> list[dict]:
        op_descriptions = format_op_descriptions(self.ctx.task_type)
        c_t = self.scheduler.current_c(step)
        user = render_user_initial(
            task_type=self.ctx.task_type,
            data_name=self.ctx.data_name,
            summary_text=self._summary_text(),
            op_descriptions=op_descriptions,
            memory_context=memory_context,
            parent_obs_text=parent_obs_text,
            parent_ops=parent_ops,
            parent_fitness=parent_fitness,
            max_chain_len=self.max_chain_len,
            expansion_idx=expansion_idx,
            k_black=self.k_black,
            step=step,
            max_iterations=self.max_iterations,
            c_t=c_t,
            ordering_hint=self._ordering_hint(),
        )
        return [
            {"role": "system", "content": SYSTEM_DATAMASTER},
            {"role": "user", "content": user},
        ]

    # ------------------------------------------------------------------
    def _ordering_hint(self) -> str:
        """Render the canonical category order with a few example ops per
        bucket, so the LLM can sort its delta chain to satisfy ``is_legal``.
        """
        from baselines.common.pipeline_constraints import (
            _TABULAR_ORDER, _REC_ORDER,
        )
        from baselines.common.operator_catalog import CATALOG
        order = _REC_ORDER if self.ctx.task_type == "rec" else _TABULAR_ORDER
        # Group operators by category, restricted to the current task type.
        by_cat: dict = {}
        for name, spec in CATALOG.items():
            if spec.task_type not in (self.ctx.task_type, "both"):
                continue
            by_cat.setdefault(spec.category, []).append(name)
        lines = []
        for i, cat in enumerate(order):
            ops = sorted(by_cat.get(cat, []))[:5]
            ops_str = ", ".join(ops) if ops else "(no ops)"
            lines.append(f"  {i+1:>2}. {cat.name}: {ops_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _extract_finding(self, response: str) -> Optional[str]:
        m = _FINDING_RE.search(response)
        if not m:
            return None
        text = m.group(1).strip()
        return text or None

    # ------------------------------------------------------------------
    def _expand_one_black_child(
        self,
        parent_node,
        memory_context: str,
        parent_obs_text: str,
        expansion_idx: int,
        step: int,
    ) -> Optional[dict]:
        """Try up to ``self.max_solution_attempts`` LLM rounds to obtain a
        legal+executable delta chain, evaluate the resulting pipeline, and
        register the new black node.

        Returns a small dict describing the expansion outcome (always
        non-None unless the LLM completely refuses to answer). The new
        :class:`NodeRecord` is added to ``self.tree`` as a side-effect.
        """
        messages = self._build_messages(
            parent_obs_text=parent_obs_text,
            parent_ops=list(parent_node.pipeline_ops),
            parent_fitness=parent_node.fitness,
            memory_context=memory_context,
            expansion_idx=expansion_idx,
            step=step,
        )

        delta_steps = None
        last_error: Optional[str] = None
        last_response: Optional[str] = None
        finding_text: Optional[str] = None

        for attempt in range(1, self.max_solution_attempts + 1):
            try:
                response = self.llm.chat(messages)
            except Exception as e:
                self.n_errors += 1
                last_error = f"LLM call failed: {e}"
                self._log(f"  attempt {attempt}: {last_error}")
                continue

            messages.append({"role": "assistant", "content": response})
            self.transcript.append({
                "role": "assistant",
                "content": response,
                "step": step,
                "expansion": expansion_idx,
                "attempt": attempt,
            })
            last_response = response
            finding_text = self._extract_finding(response) or finding_text

            action = parse_response(response)
            if action.kind != "solution" or not action.chain_str:
                last_error = (
                    "Reply missing <solution>...</solution> tag. "
                    "Please emit exactly one <solution> chain."
                )
                self.n_errors += 1
                if attempt < self.max_solution_attempts:
                    messages.append({
                        "role": "user",
                        "content": render_retry_feedback(
                            last_error,
                            self.max_solution_attempts - attempt,
                            self.max_chain_len,
                        ),
                    })
                continue

            try:
                steps = chain_to_steps(action.chain_str, self.ctx, self._rng)
            except ChainParseError as e:
                last_error = f"chain parse error: {e.message}"
                self.n_errors += 1
                if attempt < self.max_solution_attempts:
                    messages.append({
                        "role": "user",
                        "content": render_retry_feedback(
                            last_error,
                            self.max_solution_attempts - attempt,
                            self.max_chain_len,
                        ),
                    })
                continue

            if not steps:
                last_error = "<solution> chain is empty after dropping Terminate."
                self.n_errors += 1
                if attempt < self.max_solution_attempts:
                    messages.append({
                        "role": "user",
                        "content": render_retry_feedback(
                            last_error,
                            self.max_solution_attempts - attempt,
                            self.max_chain_len,
                        ),
                    })
                continue

            if len(steps) > self.max_chain_len:
                last_error = (
                    f"chain length {len(steps)} exceeds max_chain_len="
                    f"{self.max_chain_len}."
                )
                self.n_errors += 1
                if attempt < self.max_solution_attempts:
                    messages.append({
                        "role": "user",
                        "content": render_retry_feedback(
                            last_error,
                            self.max_solution_attempts - attempt,
                            self.max_chain_len,
                        ),
                    })
                continue

            tentative = Pipeline(
                steps=copy.deepcopy(parent_node.accumulated_steps) + list(steps)
            )
            if not is_legal(tentative, self.ctx.task_type):
                last_error = (
                    "Resulting pipeline violates DataMaster category-order "
                    "constraints. Reorder operators or pick different ones."
                )
                self.n_errors += 1
                if attempt < self.max_solution_attempts:
                    messages.append({
                        "role": "user",
                        "content": render_retry_feedback(
                            last_error,
                            self.max_solution_attempts - attempt,
                            self.max_chain_len,
                        ),
                    })
                continue

            delta_steps = list(steps)
            last_error = None
            break

        # ------------------------------------------------------------------
        # Evaluate (or record a failed black node)
        # ------------------------------------------------------------------
        if delta_steps is None:
            child = self.tree.add_black_child(
                parent_node.node_id,
                delta_steps=[],
                fitness=None,
                metrics={},
                error=last_error or "no usable solution from LLM",
                diagnostics={"reason": "solution_rejected"},
                findings=[finding_text] if finding_text else [],
            )
            child.is_terminal = True
            backpropagate(
                self.tree,
                child.node_id,
                self.scheduler.compute_reward(child, parent_node),
            )
            attempt_record = {
                "node_id": child.node_id,
                "parent_id": parent_node.node_id,
                "step": step,
                "expansion": expansion_idx,
                "ops": [],
                "fitness": None,
                "metrics": {},
                "error": child.error,
                "response": last_response,
            }
            self.solution_attempts.append(attempt_record)
            self._log(
                f"  expansion {expansion_idx + 1}/{self.k_black}: "
                f"FAILED ({child.error})"
            )
            return attempt_record

        # Sanity-execute on the sandbox so we can fail fast / cheaply
        sandbox_err: Optional[str] = None
        exec_res = self.sandbox.execute_chain(delta_steps)
        if not exec_res.success:
            sandbox_err = exec_res.error or "sandbox execution failed"

        accumulated = copy.deepcopy(parent_node.accumulated_steps) + list(delta_steps)
        full_pipeline = Pipeline(steps=accumulated)
        # Apply the canonical repair so the evaluator sees a fully-valid pipe.
        repair(full_pipeline, self.ctx.task_type, self.ctx)

        if sandbox_err is not None:
            child = self.tree.add_black_child(
                parent_node.node_id,
                delta_steps=delta_steps,
                fitness=None,
                metrics={},
                error=f"sandbox: {sandbox_err}",
                diagnostics={"reason": "sandbox_failure"},
                findings=[finding_text] if finding_text else [],
            )
            child.is_terminal = True
            backpropagate(
                self.tree,
                child.node_id,
                self.scheduler.compute_reward(child, parent_node),
            )
            attempt_record = {
                "node_id": child.node_id,
                "parent_id": parent_node.node_id,
                "step": step,
                "expansion": expansion_idx,
                "ops": [s.op for s in delta_steps],
                "fitness": None,
                "metrics": {},
                "error": child.error,
                "response": last_response,
            }
            self.solution_attempts.append(attempt_record)
            self._log(
                f"  expansion {expansion_idx + 1}/{self.k_black}: "
                f"sandbox failed ({sandbox_err[:80]})"
            )
            return attempt_record

        # Downstream training -> (fitness, metrics, error)
        try:
            fitness, metrics, eval_err = self.downstream_evaluator(full_pipeline)
        except Exception as e:  # pragma: no cover - safety net
            fitness, metrics, eval_err = None, {}, f"{type(e).__name__}: {e}"

        diagnostics = {
            "obs_after_delta_text": (exec_res.obs.text if exec_res.obs is not None else ""),
            "delta_ops": [s.op for s in delta_steps],
            "accumulated_after_repair": [s.op for s in full_pipeline.steps],
        }

        child = self.tree.add_black_child(
            parent_node.node_id,
            delta_steps=delta_steps,
            fitness=fitness,
            metrics=metrics,
            error=eval_err,
            diagnostics=diagnostics,
            findings=[finding_text] if finding_text else [],
        )
        # Cap depth so the tree does not run away.
        if child.depth >= self.max_depth:
            child.is_terminal = True

        reward = self.scheduler.compute_reward(child, parent_node)
        backpropagate(self.tree, child.node_id, reward)

        attempt_record = {
            "node_id": child.node_id,
            "parent_id": parent_node.node_id,
            "step": step,
            "expansion": expansion_idx,
            "ops": [s.op for s in delta_steps],
            "fitness": fitness,
            "metrics": metrics,
            "error": eval_err,
            "response": last_response,
        }
        self.solution_attempts.append(attempt_record)

        fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else "n/a"
        self._log(
            f"  expansion {expansion_idx + 1}/{self.k_black}: "
            f"node={child.node_id} ops={[s.op for s in delta_steps]} "
            f"fitness={fit_str} reward={reward:.4f}"
        )
        return attempt_record

    # ------------------------------------------------------------------
    def run(self) -> DataMasterRunResult:
        # 1. Initial sandbox reset just to populate ``ctx``-relevant data.
        try:
            self.sandbox.reset()
        except Exception as e:  # pragma: no cover - sandbox is normally robust
            self._log(f"Sandbox reset failed: {e}; using repair-only fallback.")
            pipe = Pipeline()
            repair(pipe, self.ctx.task_type, self.ctx)
            return DataMasterRunResult(
                pipeline=pipe,
                success=False,
                n_iterations=0,
                n_expansions=0,
                n_errors=1,
                best_node_id=None,
                transcript=self.transcript,
                solution_attempts=self.solution_attempts,
                tree=self.tree.to_dict(),
            )

        # 2. Root node (un-evaluated; only acts as parent for the first wave).
        if self.tree.root_id is None:
            self.tree.add_root([])
        # Give the root a phantom visit so UCB does not divide by zero when
        # we score its first children using the parent visit count.
        root = self.tree.nodes[self.tree.root_id]
        if root.visits == 0:
            root.visits = 1

        n_expansions = 0
        for step in range(self.max_iterations):
            if self.n_errors >= self.max_err_cnt:
                self._log(f"max_err_cnt reached ({self.n_errors}); stopping.")
                break
            if not self.tree.has_frontier():
                self._log("frontier empty; stopping.")
                break

            v = self.scheduler.select(self.tree, step)
            if v is None:
                break
            if v.depth >= self.max_depth:
                v.is_terminal = True
                continue

            # Re-play parent state into the sandbox.
            ok, parent_obs_text, replay_err = self._replay_parent(v.accumulated_steps)
            if not ok and replay_err:
                self._log(
                    f"step {step + 1}/{self.max_iterations} "
                    f"parent={v.node_id} replay failed: {replay_err}"
                )
                # Mark this branch terminal but keep going on other branches.
                v.is_terminal = True
                v.error = (v.error or "") + f" | parent_replay_failed: {replay_err}"
                continue

            retrieved = self.memory.retrieve(v.node_id)
            mem_text = self.memory.format_context(retrieved)

            self._log(
                f"=== step {step + 1}/{self.max_iterations} parent={v.node_id} "
                f"(depth={v.depth}, visits={v.visits}, mean_R={v.mean_reward:.4f}) ==="
            )
            for k in range(self.k_black):
                if self.n_errors >= self.max_err_cnt:
                    break
                ok, parent_obs_text_k, replay_err = self._replay_parent(
                    v.accumulated_steps
                )
                if not ok and replay_err:
                    self.n_errors += 1
                    self._log(
                        f"step {step + 1}/{self.max_iterations} "
                        f"parent={v.node_id} sibling={k} replay failed: "
                        f"{replay_err}"
                    )
                    continue
                rec = self._expand_one_black_child(
                    parent_node=v,
                    memory_context=mem_text,
                    parent_obs_text=parent_obs_text_k,
                    expansion_idx=k,
                    step=step,
                )
                if rec is not None:
                    n_expansions += 1
                    if rec.get("error") is None:
                        # Record finding tied to the new child id.
                        if "response" in rec and rec["response"]:
                            t = self._extract_finding(rec["response"])
                            if t:
                                self.memory.write_finding(rec["node_id"], t)

        # 3. Pick the best node; if none succeeded, return a repair-only pipeline.
        best = self.tree.best_node()
        if best is None:
            self._log("agent produced no successful black node; using repair fallback.")
            pipe = Pipeline()
            repair(pipe, self.ctx.task_type, self.ctx)
            return DataMasterRunResult(
                pipeline=pipe,
                success=False,
                n_iterations=self.max_iterations,
                n_expansions=n_expansions,
                n_errors=self.n_errors,
                best_node_id=None,
                transcript=self.transcript,
                solution_attempts=[
                    {k: v for k, v in a.items() if k != "response"}
                    for a in self.solution_attempts
                ],
                tree=self.tree.to_dict(),
            )

        # 4. Re-build the best pipeline and apply final repair.
        final = Pipeline(steps=copy.deepcopy(best.accumulated_steps))
        repair(final, self.ctx.task_type, self.ctx)
        legal = is_legal(final, self.ctx.task_type)
        self._log(
            f"agent done. best_node={best.node_id} fitness={best.fitness} "
            f"steps={[s.op for s in final.steps]} legal={legal}"
        )

        return DataMasterRunResult(
            pipeline=final,
            success=True,
            n_iterations=self.max_iterations,
            n_expansions=n_expansions,
            n_errors=self.n_errors,
            best_node_id=best.node_id,
            transcript=self.transcript,
            solution_attempts=[
                {k: v for k, v in a.items() if k != "response"}
                for a in self.solution_attempts
            ],
            tree=self.tree.to_dict(),
        )


__all__ = [
    "DataMasterAgent",
    "DataMasterRunResult",
]
