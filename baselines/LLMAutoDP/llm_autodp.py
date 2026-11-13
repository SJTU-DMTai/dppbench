"""Group-relative LLM strategy search; a benchmark adaptation, not SFT reproduction.

Reference: https://www.vldb.org/pvldb/vol19/p794-cheng.pdf (section 3.2).
Uses the shared operator space and validation fitness. DPS, PTS, LESS and
processed-prefix caching from the original text/SFT system are not implemented.
Only identical complete candidate evaluations are reused, within a single run.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import yaml

from baselines.common.budget import BudgetExceededError, get_active_budget
from baselines.common.config import default_config_path, load_baseline_config
from baselines.common.pipeline import Pipeline, check_pipeline, repair
from baselines.common.search_evaluator import SearchPipelineEvaluator
from baselines.common.utils import get_available_ops
from baselines.DeepPrep.llm_client import LLMClient


CONFIG_KEYS = ("llm_backend", "llm_model", "temperature", "max_tokens", "timeout",
               "n_rounds", "n_candidates", "max_pipeline_len", "eval_full", "seed",
               "fast_train", "top_p", "torch_dtype", "device_map", "enable_thinking",
               "repetition_penalty", "downstream_eval_n")


def parse_group(text, limit):
    """Reject ambiguous/truncated responses rather than treating them as no-op."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(text)
    if not isinstance(obj, dict) or type(obj.get("stop", False)) is not bool:
        raise ValueError("Expected an object with boolean stop")
    candidates = obj.get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) > limit:
        raise ValueError(f"candidates must be a list with at most {limit} entries")
    if obj.get("stop", False):
        if candidates:
            raise ValueError("stop=true requires an empty candidates list")
        return True, []
    if not candidates:
        raise ValueError("Provide at least one candidate or stop=true")
    return False, candidates


def decode_pipeline(candidate):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("dag"), dict):
        raise ValueError("Each candidate must contain a dag object")
    dag = candidate["dag"]
    if not isinstance(dag.get("sources"), list) or not dag["sources"]:
        raise ValueError("dag.sources must be a nonempty list")
    if not isinstance(dag.get("ops"), list) or not isinstance(dag.get("train"), dict):
        raise ValueError("dag.ops and dag.train are required")
    return Pipeline.from_yaml(yaml.safe_dump({"dag": dag}, sort_keys=False))


class LLMAutoDP:
    """LLMAutoDP: compare groups against a mandatory-only anchor."""

    def __init__(self, task_dir, data_name, data_dir=None, output_dir=None,
                 seed=None, verbose=True, device="cpu", model_name=None,
                 eval_full=None, max_pipeline_len=None, config_path=None,
                 llm_backend=None, llm_model=None):
        self.cfg = load_baseline_config(config_path or default_config_path(__file__), CONFIG_KEYS)
        for key, value in {"seed": seed, "eval_full": eval_full,
                           "max_pipeline_len": max_pipeline_len,
                           "llm_backend": llm_backend, "llm_model": llm_model}.items():
            if value is not None:
                self.cfg[key] = value
        for key in ("n_rounds", "n_candidates", "max_pipeline_len", "max_tokens", "timeout"):
            if not isinstance(self.cfg[key], int) or self.cfg[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.verbose, self.device, self.model_name = verbose, device, model_name
        self.output_dir = Path(output_dir or Path("outputs") / "LLMAutoDP" / data_name)

    def _make_evaluator(self):
        evaluator = SearchPipelineEvaluator(
            task_dir=self.task_dir, data_name=self.data_name, data_dir=self.data_dir,
            small_n=self.cfg["downstream_eval_n"], seed=self.cfg["seed"], verbose=self.verbose,
            device=self.device, model_name=self.model_name)
        evaluator.set_evaluate_std_test(False)
        evaluator.set_fast_mode(bool(self.cfg["fast_train"]))
        return evaluator

    def _make_llm(self):
        return LLMClient(backend=self.cfg["llm_backend"], model=self.cfg["llm_model"],
                         temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"],
                         timeout=self.cfg["timeout"], top_p=self.cfg["top_p"],
                         torch_dtype=self.cfg["torch_dtype"], device_map=self.cfg["device_map"],
                         enable_thinking=self.cfg["enable_thinking"],
                         repetition_penalty=self.cfg["repetition_penalty"])

    def _search(self, evaluator, llm, ctx):
        # No task prepare.yaml is loaded. Required structural operators are the
        # benchmark equivalent of the paper's unprocessed-data anchor.
        anchor = repair(Pipeline(), ctx.task_type, ctx, rng=random.Random(self.cfg["seed"]))
        history, cache = [], {}
        best, best_fit, best_metrics = None, None, {}
        allowed_ops = {spec.op_name for spec in get_available_ops(ctx.task_type)}

        def score(pipeline):
            nonlocal best, best_fit, best_metrics
            if any(step.op not in allowed_ops for step in pipeline.steps):
                return {"fitness": None, "error": "Operator outside the task whitelist"}
            report = check_pipeline(pipeline, ctx)
            if not report.valid:
                return {"fitness": None, "error": "; ".join(report.errors)}
            if len(pipeline.steps) > self.cfg["max_pipeline_len"]:
                return {"fitness": None, "error": "max_pipeline_len exceeded"}
            key = pipeline.hash()
            if key in cache:
                return {**cache[key], "cache_hit": True}
            ev = evaluator.evaluate(pipeline)
            if ev.error and "BudgetExceededError" in ev.error:
                raise BudgetExceededError(ev.error)
            fit = float(ev.fitness) if ev.success and math.isfinite(ev.fitness) else None
            record = {"fitness": fit, "error": ev.error, "cache_hit": False}
            cache[key] = record
            if fit is not None and (best_fit is None or fit > best_fit):
                best, best_fit, best_metrics = pipeline.copy(), fit, dict(ev.metrics or {})
            return record

        anchor_result = score(anchor)
        r0 = anchor_result["fitness"]
        history.append({"round": 0, "anchor": yaml.safe_load(anchor.to_yaml()), **anchor_result})
        catalog = [asdict(spec) for spec in get_available_ops(ctx.task_type)]
        system = (
            "You are LLMAutoDP, searching data preparation strategies for DPPBench. "
            "Generate multiple diverse complete strategies, compare each group's validation "
            "fitness and improvement relative to the fixed anchor, then revise operator choices "
            "and order. Higher fitness is better (error metrics are already negated). "
            "The anchor contains only benchmark-required preparation, not a reference solution. "
            "Use only the supplied operators and valid parameters; preserve target, ID, time, "
            "split columns, recommendation histories and graph node/edge alignment. "
            "Never use test data or change labels. Do not generate code. "
            "Return JSON only: {\"stop\": false, \"candidates\": [{\"dag\": "
            "{\"sources\": [{\"id\": \"s0\", \"table\": \"main\"}], "
            "\"ops\": [{\"id\": \"o1\", \"op\": \"HandleMV\", \"prev\": [\"s0\"], "
            "\"params\": {}}], \"train\": {\"prev\": [\"o1\"]}}}]}. "
            "Sources must use this task's table names; retain mandatory operators. "
            f"Return 1 to {self.cfg['n_candidates']} candidates per round; at most "
            f"{self.cfg['max_pipeline_len']} operators per candidate. Start with simple diverse "
            "strategies. To stop and select the highest-scoring evaluated strategy return "
            "{\"stop\": true, \"candidates\": []}. No untested final strategies."
        )
        messages = [{"role": "user", "content": json.dumps({
            "context": asdict(ctx), "operators": catalog, "anchor": history[0],
            "rounds": self.cfg["n_rounds"]}, default=str)}]
        stop_reason = "round_limit"
        for turn in range(1, self.cfg["n_rounds"] + 1):
            try:
                budget = get_active_budget()
                if budget:
                    budget.check_resources()
                response = llm.chat(messages, system=system)
                messages.append({"role": "assistant", "content": response})
                try:
                    stop, candidates = parse_group(response, self.cfg["n_candidates"])
                except (ValueError, TypeError) as exc:
                    feedback = {"round": turn, "error": str(exc)}
                else:
                    if stop:
                        stop_reason = "llm_stop"
                        break
                    group = []
                    for candidate in candidates:
                        try:
                            pipeline = decode_pipeline(candidate)
                            result = score(pipeline)
                        except BudgetExceededError:
                            raise
                        except (ValueError, TypeError, KeyError, AttributeError) as exc:
                            result = {"fitness": None, "error": str(exc)}
                        fit = result["fitness"]
                        group.append({"candidate": candidate, **result,
                                      "relative_improvement": fit - r0 if fit is not None and r0 is not None else None})
                    feedback = {"round": turn, "group": group, "anchor_fitness": r0,
                                "best_fitness": best_fit}
                history.append(feedback)
                messages.append({"role": "user", "content": json.dumps(feedback, default=str)})
                if self.verbose:
                    print(f"[LLMAutoDP] round={turn} best={best_fit}", flush=True)
            except BudgetExceededError as exc:
                stop_reason = str(exc)
                break
        if best is None:
            raise RuntimeError("LLM-AutoDP produced no successful validation candidate: " + json.dumps(history, default=str))
        return best, best_fit, best_metrics, history, messages, stop_reason

    def run(self):
        started = time.monotonic()
        evaluator = self._make_evaluator()
        try:
            ctx = evaluator._get_data_context()
            best, fit, metrics, history, messages, stop_reason = self._search(evaluator, self._make_llm(), ctx)
        finally:
            evaluator._executor.cleanup()
        final = None
        if self.cfg["eval_full"]:
            evaluator = self._make_evaluator()
            try:
                evaluator.set_small_n(0)
                evaluator.set_fast_mode(False)
                final = evaluator.evaluate(best)
            finally:
                evaluator._executor.cleanup()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = self.output_dir / "best_pipeline.yaml"
        pipeline_path.write_text(best.to_yaml(), encoding="utf-8")
        summary = {
            "method": "LLMAutoDP", "original_paper_reproduction": False,
            "data_name": self.data_name, "task_type": ctx.task_type,
            "best_pipeline_yaml": best.to_yaml(), "best_pipeline_path": str(pipeline_path.resolve()),
            "best_fitness_in_loop": fit, "best_metrics_in_loop": metrics,
            "best_fitness": final.fitness if final and final.success else None,
            "best_metrics": final.metrics if final else {}, "eval_error": final.error if final else None,
            "is_legal": True, "agent_success": True, "stop_reason": stop_reason,
            "duration_seconds": time.monotonic() - started,
            "adaptations": {"DPS": False, "PTS": False, "LESS": False,
                            "processed_prefix_cache": False, "full_candidate_cache": True},
            "config": self.cfg,
        }
        for name, value in (("run_summary.json", summary), ("trajectory.json", history),
                            ("agent_log.json", messages)):
            (self.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return summary
