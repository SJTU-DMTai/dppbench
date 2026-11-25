"""Validation-only fixed references under the original DefaultCredit protocol.

This is an additional diagnostic, not a replacement baseline: native NaNs may
violate benchmark mandatory-imputation rules, so reference training deliberately
uses the unchanged TrainingExecutor without baseline search/legality gating.
Original loaders, per-seed splits, operators, model and final HPO are reused.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from baselines.common.executor import TrainingExecutor
from baselines.common.evaluator import PipelineEvaluator, EvaluationResult
from baselines.common.pipeline import Pipeline
from scripts.final_model_hpo import tune_final_model, candidate_configs
from scripts.default_credit_missingness import sha, disjoint, dump

METHODS = ["native", "mean", "median", "constant", "knn", "iterative"]
HISTORY = [f"{field}_{agg}" for field in ("PAY_STATUS", "BILL_AMT", "PAY_AMT")
           for agg in ("mean", "max", "min")]
FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            *HISTORY, "MONTH_OFFSET_mean", "MONTH_OFFSET_max", "MONTH_OFFSET_min"]


def reference_pipeline(method):
    if method not in METHODS:
        raise ValueError(method)
    # Fixed, explicit graph: the secondary table is bound by prev, not aux_df.
    ops = [{"id": "o1", "op": "JoinTable", "prev": ["s0", "s1"],
            "params": {"method": "agg", "key_col": "ID", "how": "left",
                       "agg_funcs": ["mean", "max", "min"]}},
           {"id": "o2", "op": "LabelEncode", "prev": ["o1"]}]
    if method != "native":
        params = {"method": method, "action": "impute", "cols": FEATURES}
        if method == "constant":
            params["fill_value"] = 0
        ops.append({"id": "o3", "op": "HandleMV", "prev": ["o2"], "params": params})
    return Pipeline.from_yaml(yaml.safe_dump({"dag": {
        "sources": [{"id": "s0", "table": "main"},
                    {"id": "s1", "table": "monthly_history"}],
        "ops": ops, "train": {"prev": [ops[-1]["id"]]}}}))


def frame_audit(frame):
    if not set(HISTORY) <= set(frame.columns):
        raise ValueError(f"Historical aggregate columns missing: {set(HISTORY)-set(frame.columns)}")
    result = {}
    # Audit and score ONLY original training/validation rows. No test metrics.
    for split in ("train", "val"):
        part = frame.loc[frame["__split__"].eq(split)]
        if part.empty or not part.ID.is_unique:
            raise ValueError(f"Invalid {split} membership")
        result[split] = {"n_rows": len(part),
                         "id_sha256": __import__('hashlib').sha256(
                             pd.util.hash_pandas_object(part.ID, index=False).values.tobytes()).hexdigest(),
                         "missing_history": {col: int(part[col].isna().sum()) for col in HISTORY},
                         "missing_model_features": int(part[FEATURES].isna().sum().sum())}
    return result


class AuditExecutor(TrainingExecutor):
    """Observe original instance preprocessing; do not override data/splits."""
    def _make_configured_data_instance(self):
        data = super()._make_configured_data_instance()
        transform = data._transform_tabular_dag_df

        def audited_transform(frame, op, op_name):
            if op_name == "LabelEncode":
                self.diagnostic_audit["before"] = frame_audit(frame)
            if op_name == "HandleMV":
                if set(op.cols or []) != set(FEATURES):
                    raise ValueError("Reference imputer must explicitly cover every model predictor")
                if any(col in op.cols for col in ("ID", "TARGET", "__split__")):
                    raise ValueError("Reference imputer consumed protected column")
            result = transform(frame, op, op_name)
            if op_name == "HandleMV" and op.method != self.reference_method:
                raise ValueError(f"Imputer silently fell back: {self.reference_method} -> {op.method}")
            return result

        data._transform_tabular_dag_df = audited_transform
        preprocess = data.run_pre_process

        def audited_preprocess(path):
            preprocess(path)
            after = frame_audit(data.train_df)
            self.diagnostic_audit["after"] = after
            before = self.diagnostic_audit["before"]
            for split in ("train", "val"):
                if before[split]["id_sha256"] != after[split]["id_sha256"]:
                    raise ValueError("Reference changed original partition row order/membership")
                if self.reference_method != "native" and after[split]["missing_model_features"]:
                    raise ValueError("Imputation did not remove all model-feature missing values")
                if self.reference_method == "native" and before[split]["missing_history"] != after[split]["missing_history"]:
                    raise ValueError("Native reference unexpectedly removed history missing values")
        data.run_pre_process = audited_preprocess
        return data


class ReferenceEvaluator(PipelineEvaluator):
    executor_class = AuditExecutor
    audit_records = []

    def evaluate(self, pipeline, scoring_method=None):
        """Fixed-reference path only. Does not change baseline legality rules."""
        if self._executor._evaluate_std_test:
            raise ValueError("Reference diagnostic must disable standard-test scoring")
        steps = [step for step in pipeline.steps if step.op == "HandleMV"]
        method = steps[0].params["method"] if steps else "native"
        self._executor.reference_method = method
        self._executor.diagnostic_audit = {}
        self._executor.write_yaml(pipeline.to_yaml())
        started = time.monotonic()
        result = self._executor.run_training()
        metrics = result.get("metrics") or {}
        if any(key.startswith("std_test_") for key in metrics):
            raise ValueError("Unexpected standard-test metric in validation diagnostic")
        auc = metrics.get("auc")
        success = bool(result.get("success") and auc is not None and math.isfinite(float(auc)))
        self.audit_records.append({"method": method, "model_config": self._executor.get_model_config(),
                                   "success": success, "error": result.get("error"),
                                   "audit": self._executor.diagnostic_audit})
        return EvaluationResult(float(auc) if success else -math.inf, success,
                                metrics, result.get("error"), time.monotonic()-started)


def input_fingerprints(root):
    task = Path(root) / "default_credit"
    xls = task / "data/default of credit card clients.xls"
    cache = Path(str(xls) + ".pkl")
    files = [xls, cache, *(task / "std_test" / name for name in
                          ("train_frozen.parquet", "std_test.parquet", "meta.json"))]
    if not all(path.is_file() for path in files):
        raise ValueError(f"Require existing XLS/cache/frozen artifacts under {task}; no download/build allowed")
    if cache.stat().st_mtime < xls.stat().st_mtime:
        raise ValueError(f"Cache would be regenerated in the data root; refusing: {cache}")
    return {str(path): sha(path) for path in files}


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams: stream.flush()


def diagnose_one(root, seed, method, output):
    output.mkdir(parents=True, exist_ok=False)
    pipeline = reference_pipeline(method)
    (output / "reference.yaml").write_text(pipeline.to_yaml(), encoding="utf-8")
    start = len(ReferenceEvaluator.audit_records)
    evaluator = None
    try:
        evaluator, summary = tune_final_model(
            pipeline, output_dir=str(output), evaluator_class=ReferenceEvaluator,
            task_dir=str(REPO / "dppbench/tasks/default_credit"), data_name="default_credit",
            data_dir=str(root), model_name="LightGBM", seed=seed, device="cpu", verbose=True)
        # tune_final_model returns a fresh evaluator for final testing. We do
        # NOT run it: best validation score is already recorded by the trials.
        successful = [trial for trial in summary["trials"] if trial["success"]]
        best = max(successful, key=lambda trial: float(trial["value"]))
        record = {"success": len(successful) == len(summary["trials"]),
                  "method": method, "seed": seed, "best_validation_auc": float(best["value"]),
                  "best_params": summary["best_params"], "hpo_seconds": summary["seconds"],
                  "successful_trials": len(successful), "n_trials": len(summary["trials"]),
                  "evaluation": "validation_only_no_final_test_call",
                  "reference_only_not_baseline_search": True,
                  "diagnostic_audits": ReferenceEvaluator.audit_records[start:]}
        dump(output / "validation.json", record)
        if not record["success"]:
            raise RuntimeError(f"Some HPO trials failed: {output / 'final_hpo.json'}")
        return record
    finally:
        if evaluator is not None:
            evaluator._executor.cleanup()


def parse_roots(items):
    roots = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in roots:
            raise ValueError("Use unique NAME=/absolute/data/root arguments")
        roots[name] = Path(value).resolve()
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", required=True, help="clean=/original/root mask1001=/perturbed/root")
    parser.add_argument("--output-root", required=True, help="NEW standalone diagnostic output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--execute", action="store_true", help="Otherwise print plan without training or writing")
    args = parser.parse_args()
    roots = parse_roots(args.data_roots)
    output = Path(args.output_root).resolve()
    disjoint(output, REPO)
    for root in roots.values(): disjoint(output, root)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate seeds or methods")
    trials = len(candidate_configs("LightGBM"))
    print(f"Plan: {len(roots)} datasets x {len(args.seeds)} seeds x {len(args.methods)} references x {trials} HPO trials")
    print("Original loaders and per-seed splits; CPU LightGBM; validation scores only; no baseline/config modifications.")
    if not args.execute:
        return
    if output.exists(): raise FileExistsError(f"Use a new diagnostic output directory: {output}")
    fingerprints = {name: input_fingerprints(root) for name, root in roots.items()}
    output.mkdir(parents=True)
    dump(output / "input_hashes.json", fingerprints)
    dump(output / "plan.json", {"roots": {name:str(root) for name,root in roots.items()},
                               "seeds":args.seeds,"methods":args.methods,
                               "HPO_candidates":candidate_configs("LightGBM"),
                               "source_script_sha256":sha(__file__),
                               "protocol":"original_loader_splits_final_HPO_validation_only"})
    records, failures = [], []
    try:
        for name, root in roots.items():
            for seed in args.seeds:
                for method in args.methods:
                    target = output / name / f"seed{seed}" / method
                    target.parent.mkdir(parents=True, exist_ok=True)
                    log_path = target.parent / f"{method}.log"
                    with log_path.open("x", encoding="utf-8") as log:
                        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                            print(f"[START] dataset={name} seed={seed} reference={method}", flush=True)
                            try:
                                record = diagnose_one(root, seed, method, target)
                                record["dataset"] = name
                                records.append(record)
                                print(f"[SUCCESS] best validation AUC={record['best_validation_auc']:.6f}", flush=True)
                            except Exception as exc:
                                traceback.print_exc()
                                failures.append({"dataset":name,"seed":seed,"method":method,"error":str(exc)})
                                dump(target.parent / f"{method}_error.json", failures[-1])
                    dump(output / "summary.json", records)
                    dump(output / "summary_brief.json", [
                        {key: record[key] for key in ("dataset", "seed", "method", "success",
                         "best_validation_auc", "hpo_seconds", "successful_trials", "n_trials")}
                        for record in records])
                    dump(output / "failures.json", failures)
    finally:
        for name, root in roots.items():
            if input_fingerprints(root) != fingerprints[name]:
                raise RuntimeError(f"Dataset inputs changed: {name}")
        print("Verified: original and corrupted input files unchanged", flush=True)
    if failures: raise SystemExit(f"{len(failures)} references failed; inspect failures.json and per-reference logs")


if __name__ == "__main__": main()
