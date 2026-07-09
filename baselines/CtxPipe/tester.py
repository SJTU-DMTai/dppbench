"""Greedy inference for component-level CtxPipe."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from baselines.common.pipeline import Pipeline
    from baselines.common.search_evaluator import SearchPipelineEvaluator

    from .agent import DQNAgent
    from .env import PipelineEnv


class CtxPipeTester:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = bool(verbose)

    def run(
        self,
        agent: "DQNAgent",
        env: "PipelineEnv",
        eval_full_with: Optional["SearchPipelineEvaluator"] = None,
    ) -> dict:
        state = env.reset()
        done = False
        rl_reward = 0.0
        rl_info: dict = {}
        trace: list[dict] = []

        while not done:
            component = env.current_component_id()
            if component is None:
                break
            mask = env.action_mask()
            action = agent.select_action(component, state, eps=0.0, action_mask=mask)
            next_state, reward, done, info = env.step(action)
            trace.append({
                "component": component,
                "category": info.get("category"),
                "action": int(action),
                "selected": info.get("op_added") or info.get("action_name"),
                "reason": info.get("reason"),
                "reward": float(reward),
            })
            state = next_state
            rl_reward += float(reward)
            if done:
                rl_info = info

        pipeline: "Pipeline" = env.current_pipeline()
        out = {
            "pipeline": pipeline,
            "pipeline_yaml": pipeline.to_yaml(),
            "ops": [s.op for s in pipeline.steps],
            "rl_reward": rl_reward,
            "rl_success": bool(rl_info.get("success", False)),
            "rl_fitness": rl_info.get("fitness"),
            "rl_metrics": rl_info.get("metrics") or {},
            "rl_error": rl_info.get("error"),
            "trace": trace,
        }

        if eval_full_with is not None:
            if self.verbose:
                print(f"[CtxPipe] re-evaluating final pipeline on full data "
                      f"({len(pipeline)} ops)")
            full_ev = eval_full_with.evaluate(pipeline)
            out["full_success"] = bool(full_ev.success)
            out["full_fitness"] = float(full_ev.fitness) if full_ev.success else None
            out["full_metrics"] = dict(full_ev.metrics or {})
            out["full_error"] = full_ev.error
            out["fitness"] = out["full_fitness"]
            out["metrics"] = out["full_metrics"]
            out["error"] = full_ev.error
        else:
            out["fitness"] = out["rl_fitness"]
            out["metrics"] = out["rl_metrics"]
            out["error"] = out["rl_error"]

        if self.verbose:
            fit = out.get("fitness")
            fit_str = f"{fit:.4f}" if isinstance(fit, float) else "n/a"
            print(f"[CtxPipe] final ops={out['ops']}  fitness={fit_str}")

        return out
