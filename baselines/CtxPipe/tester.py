"""Greedy inference using CtxPipe's logical and component policies."""
from __future__ import annotations


class CtxPipeTester:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = bool(verbose)

    def run(self, agent, env, eval_full_with=None) -> dict:
        env.reset(0)
        logical_id, _ = agent.select_logical(
            env.logical_state(), env.context_embedding(), epsilon=0.0,
            context_factory=env.context_embedding,
        )
        env.set_logical_pipeline(logical_id)
        trace = []
        info = {}
        while not env.done:
            component = env.current_component_id()
            state = env.get_state()
            context = env.context_embedding()
            tried = list(env.invalid_actions(component))
            while True:
                action, _ = agent.select_component(
                    component, state, context, epsilon=0.0, tried=tried,
                    context_factory=env.context_embedding,
                )
                tried.append(action)
                _, reward, done, info = env.step(action)
                if not info.get("retry") or len(tried) >= env.component(component).action_dim:
                    break
            trace.append({
                "component": component, "action": int(action),
                "selected": info.get("op_added") or info.get("action_name"),
                "reward": float(reward), "error": info.get("error"),
            })
            if info.get("retry"):
                break

        pipeline = env.current_pipeline()
        out = {
            "pipeline": pipeline, "pipeline_yaml": pipeline.to_yaml(),
            "ops": pipeline.op_names(), "rl_reward": info.get("fitness", reward),
            "rl_success": bool(info.get("success", False)),
            "rl_fitness": info.get("fitness"),
            "rl_metrics": dict(info.get("metrics") or {}),
            "rl_error": info.get("error"), "trace": trace,
            "logical_pipeline_id": logical_id,
            "logical_order": list(env.logical_pipelines[logical_id]),
        }
        if eval_full_with is not None:
            result = eval_full_with.evaluate(pipeline)
            out.update({
                "fitness": float(result.fitness) if result.success else None,
                "metrics": dict(result.metrics or {}), "error": result.error,
                "full_success": bool(result.success),
            })
        else:
            out.update({
                "fitness": out["rl_fitness"], "metrics": out["rl_metrics"],
                "error": out["rl_error"],
            })
        if self.verbose:
            print(
                f"[CtxPipe] logical_pipeline={logical_id} "
                f"order={out['logical_order']} ops={out['ops']} fitness={out['fitness']}"
            )
        return out


__all__ = ["CtxPipeTester"]
