"""Prepare a shared, budget-matched, Random-victim SAVAGE-style MAR experiment."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import prepare_multidata_missingness as base

FIELDS = {
    "polish_bankruptcy": ["Attr3", "Attr4", "Attr7"],
    "home_credit": ["DAYS_ID_PUBLISH", "DAYS_REGISTRATION", "CNT_CHILDREN"],
    "fraud_detection": ["TransactionDT", "card1", "card2"],
}
PARTS = ("train_frozen", "std_test")
# Canonical pipeline fingerprints recorded in exp1_10_30/seed42/*_Random_seed42.log.
CLEAN_RANDOM = {
    "fraud_detection": "406868f5e48347e5d519beeed1135b633208accbeb12e1183a50926ae0642bb7",
    "home_credit": "947e52a8c2745472c5adf4e8e33a9d9783e04d5eb79f782c5a3f8d25f8977e59",
    "polish_bankruptcy": "051be943e24b855f41ab8f1a02bfe40ca236dcd0e181e2415f98600531af3c86",
}


def victim_digest(path):
    from baselines.common.pipeline import Pipeline
    canonical = Pipeline.from_yaml(path.read_text(encoding="utf-8")).to_yaml()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InsufficientSupport(ValueError):
    pass


def exact_mask(frame, cols, rule, partitions, budgets, seed, task):
    eligible = np.ones(len(frame), dtype=bool)
    for field, high, threshold in rule:
        values = frame[field].to_numpy(dtype=float)
        eligible &= np.isfinite(values) & ((values > threshold) if high else (values <= threshold))
    mask = pd.DataFrame(False, index=frame.index, columns=cols)
    for j, col in enumerate(cols):
        observed = eligible & frame[col].notna().to_numpy()
        rng = np.random.default_rng(np.random.SeedSequence([seed, list(base.TASKS).index(task), j]))
        priority = rng.random(len(frame))
        for indices, counts in zip(partitions, budgets):
            candidates = indices[observed[indices]]
            k = int(counts[col])
            if len(candidates) < k:
                raise InsufficientSupport(f"{col}: {len(candidates)} eligible for {k} cells")
            if k:
                selected = candidates[np.argpartition(priority[candidates], k-1)[:k]]
                mask.iloc[selected, j] = True
    return mask


def apply_mask(frame, mask):
    out = frame.copy()
    for col in mask:
        if mask[col].any():
            out.loc[mask[col], col] = np.nan
    return out


def inner_data(pool, task, limit):
    cfg = base.TASKS[task]
    y = pool[cfg["target"]]
    ratio = base.yaml_config(task)["train"]["val_ratio"]
    outer, _ = train_test_split(np.arange(len(pool)), test_size=ratio, stratify=y, random_state=42)
    if len(outer) > limit:
        outer, _ = train_test_split(outer, train_size=limit, stratify=y.iloc[outer], random_state=1001)
    fit, dev = train_test_split(np.arange(len(outer)), test_size=ratio, stratify=y.iloc[outer], random_state=42)
    return outer, (fit, dev)


def make_victim(task, stage, pipeline):
    from baselines.common.executor import TrainingExecutor

    # Load auxiliaries only from the private staging copy. Each trial receives
    # fresh mutable data; even the original unlabelled competition test is absent.
    raw = base.load_class(base.TASKS[task]["class"])(data_dir=str(stage/task/"data"))
    raw.load_data()
    aux = raw.auxiliary_dfs
    keys = getattr(raw, "auxiliary_join_keys", {})
    del raw

    class Victim(TrainingExecutor):
        def _make_configured_data_instance(self):
            data = super()._make_configured_data_instance()
            data.auxiliary_dfs = {k: v.copy(deep=True) for k, v in aux.items() if v is not None}
            data.auxiliary_join_keys = copy.deepcopy(keys)
            data.train_df = self.attack_frame.copy()
            data.test_df = None
            data._skip_std_test_loading = True
            data.load_data = lambda: (data.train_df, None)
            return data

    executor = Victim(str(base.REPO/"dppbench/tasks"/task), task, str(stage),
                      device="cpu", model_name="LightGBM", seed=42)
    executor.set_evaluate_std_test(False)
    executor.write_yaml(pipeline.read_text(encoding="utf-8"))
    return executor


def score(executor, frame):
    random.seed(42)
    np.random.seed(42)
    executor.attack_frame = frame
    result = executor.run_training()
    if not result["success"]:
        raise RuntimeError(result["error"])
    auc = float(result["metrics"]["auc"])
    if not np.isfinite(auc) or any(k.startswith("std_test") for k in result["metrics"]):
        raise ValueError("Invalid attack-development score")
    return auc


def search(task, frame, parts, counts, evaluate, args, folder):
    import optuna
    fields = FIELDS[task]
    fit = frame.iloc[parts[0]]
    cols = base.TASKS[task]["cols"]
    parents = [([], 1.)]
    best = None
    study_id = 0
    started = time.monotonic()
    with (folder/"trials.jsonl").open("x", encoding="utf-8") as log:
        for depth in (1, 2):
            children = []
            for parent, _ in parents:
                for field in fields:
                    if any(p[0] == field for p in parent):
                        continue
                    finite = fit[field].to_numpy(dtype=float)
                    finite = finite[np.isfinite(finite)]
                    if not len(finite) or np.min(finite) == np.max(finite):
                        continue
                    for high in (False, True):
                        study_id += 1
                        sampler = optuna.samplers.TPESampler(seed=args.mask_seed+study_id, n_startup_trials=3)
                        study = optuna.create_study(direction="minimize", sampler=sampler)
                        local = None
                        for _ in range(args.trials):
                            trial = study.ask()
                            q = trial.suggest_float("excluded_tail", .05, .45)
                            threshold = float(np.quantile(finite, q if high else 1-q))
                            rule = parent + [(field, high, threshold)]
                            record = dict(depth=depth, rule=rule, quantile_tail=q, study=study_id)
                            try:
                                mask = exact_mask(frame, cols, rule, parts, counts, args.mask_seed, task)
                            except InsufficientSupport as exc:
                                record["infeasible"] = str(exc)
                                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                            else:
                                auc = evaluate(apply_mask(frame, mask))
                                study.tell(trial, auc)
                                record["dev_auc"] = auc
                                if local is None or auc < local[1]:
                                    local = (rule, auc)
                            record["elapsed_seconds"] = time.monotonic()-started
                            log.write(json.dumps(record)+"\n")
                            log.flush()
                        if local is not None:
                            children.append(local)
            if not children:
                break
            parents = sorted(children, key=lambda item: item[1])[:args.beam]
            if best is None or parents[0][1] < best[1]:
                best = parents[0]
    if best is None:
        raise ValueError("No feasible MAR candidate")
    return best


def run_task(args, task, source, mcar, stage, output):
    cfg = base.TASKS[task]
    cols = cfg["cols"]
    folder = output/"attack"/task
    folder.mkdir(parents=True)
    original_hashes = json.loads((mcar/task/"audit/source_hashes.json").read_text())
    for sub, names in [("data", cfg["files"]), ("std_test", ["meta.json", *(p+".parquet" for p in PARTS)])]:
        dst = stage/task/sub
        dst.mkdir(parents=True)
        for name in names:
            src = source/task/sub/name
            if src.is_symlink() or not src.is_file():
                raise ValueError(f"Require regular input: {src}")
            shutil.copyfile(src, dst/name)
            if base.sha(dst/name) != original_hashes[str(src)]:
                raise ValueError(f"Source differs from original MCAR experiment: {src}")
    yaml_path = Path(args.random_root)/f"{task}_LightGBM/best_pipeline.yaml"
    frozen_yaml = folder/"random_seed42.yaml"
    shutil.copyfile(yaml_path, frozen_yaml)
    if victim_digest(frozen_yaml) != CLEAN_RANDOM[task]:
        raise ValueError(f"Victim differs from original clean Random seed42: {yaml_path}")
    pool = pd.read_parquet(stage/task/"std_test/train_frozen.parquet").drop(columns="__split__", errors="ignore")
    outer, parts = inner_data(pool, task, args.max_attack_rows)
    frame = pool.iloc[outer].reset_index(drop=True)
    del pool
    sampled = pd.read_parquet(mcar/task/"std_test/train_frozen.parquet", columns=cols).iloc[outer].reset_index(drop=True)
    mcar_mask = sampled.isna() & frame[cols].notna()
    pd.testing.assert_frame_equal(apply_mask(frame[cols], mcar_mask), sampled, check_dtype=False, check_exact=True)
    counts = [mcar_mask.iloc[idx].sum().astype(int).to_dict() for idx in parts]
    base.dump(folder/"development.json", {"fit_pool_rows": outer[parts[0]].tolist(),
              "dev_pool_rows": outer[parts[1]].tolist(), "injected_counts": counts})
    executor = make_victim(task, stage, frozen_yaml)
    started = time.monotonic()
    try:
        clean_auc = score(executor, frame)
        mcar_auc = score(executor, apply_mask(frame, mcar_mask))
        rule, auc = search(task, frame, parts, counts, lambda df: score(executor, df), args, folder)
        base.dump(folder/"selected.json", dict(rule=rule, clean_dev_auc=clean_auc, mcar_dev_auc=mcar_auc,
            selected_dev_auc=auc, search_seconds=time.monotonic()-started, model=executor.get_model_config(),
            victim_source=str(yaml_path), victim_sha256=base.sha(frozen_yaml)))
    finally:
        executor.cleanup()
    del executor, frame, sampled
    # Only after committing the rule, obtain the final per-partition budgets.
    def mask_callback(raw, columns, rate, seed, task):
        budgets = []
        frozen = [pd.read_parquet(stage/task/"std_test"/(p+".parquet")) for p in PARTS]
        indices = base.partition_positions(raw, *frozen, cfg)
        for part, clean in zip(PARTS, frozen):
            ref = pd.read_parquet(mcar/task/"std_test"/(part+".parquet"), columns=cols)
            budgets.append((ref.isna() & clean[cols].notna()).sum().to_dict())
        mask = exact_mask(raw, columns, rule, indices, budgets, seed, task)
        return apply_mask(raw, mask), mask
    with patch.object(base, "mask_frame", mask_callback):
        base.prepare_task(stage, output, task, .3, args.mask_seed)
    base.dump(output/task/"audit/source_hashes.json", original_hashes)


ACCEPTED_MCAR_DRIFT = {
    "baselines/CtxPipe/env.py", "baselines/CtxPipe/config.yaml",
    "baselines/CtxPipe/context.py", "baselines/CtxPipe/trainer.py",
    "baselines/FlowPipe/config.yaml", "baselines/DAGPipe/config.yaml",
    "baselines/HAIPipe/config.yaml", "scripts/run_full_baseline_experiments.py",
}


def verify_mcar(root):
    if (root/"INCOMPLETE.json").exists():
        raise ValueError("Incomplete MCAR preparation")
    manifest = json.loads((root/"missingness_manifest.json").read_text())
    if Path(manifest["repo"]).resolve() != base.REPO.resolve():
        raise ValueError("Verify against the original experiment checkout")
    hashes = manifest["code_hashes"]
    drift = []
    for rel, expected in hashes.items():
        path = base.REPO/rel
        current = base.sha(path) if path.is_file() else None
        if current != expected:
            if current is None or Path(rel).as_posix() not in ACCEPTED_MCAR_DRIFT:
                drift.append(rel)
            else:
                print(f"[ACCEPTED MCAR DRIFT] {rel}: {expected} -> {current}", flush=True)
    if drift:
        raise ValueError("Other tracked code/config differs from MCAR:\n" + "\n".join(drift))
    for rel, expected in manifest["input_hashes"].items():
        if base.sha(root/rel) != expected:
            raise ValueError(f"MCAR input changed: {rel}")
    for task in manifest["tasks"]:
        for path, expected in json.loads((root/task/"audit/source_hashes.json").read_text()).items():
            if base.sha(path) != expected:
                raise ValueError(f"Original input changed: {path}")
    print("[VERIFIED] MCAR inputs, original inputs and in-scope code/config", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "mcar-root", "random-root", "output-root"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--tasks", nargs="+", choices=list(FIELDS), default=list(FIELDS))
    p.add_argument("--mask-seed", type=int, default=1001)
    p.add_argument("--max-attack-rows", type=int, default=12000)
    p.add_argument("--trials", type=int, default=6, help="TPE trials per parent/field/direction")
    p.add_argument("--beam", type=int, default=2)
    args = p.parse_args()
    if args.mask_seed < 0 or min(args.trials,args.beam,args.max_attack_rows)<1 or len(set(args.tasks))!=len(args.tasks):
        p.error("Invalid seed, budget or duplicate tasks")
    source, mcar, output = [Path(x).resolve() for x in (args.source_root,args.mcar_root,args.output_root)]
    for root in (source, mcar, Path(args.random_root), base.REPO):
        base.disjoint(root, output)
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    import optuna
    import lightgbm
    manifest = verify_mcar(mcar)
    if (manifest["version"] != "original_protocol_primary_MCAR_v1" or manifest["rate"] != .3
            or manifest["mask_seed"] != 1001 or Path(manifest["source"]).resolve() != source
            or not set(args.tasks)<=set(manifest["tasks"])):
        raise ValueError("Require the matching original MCAR30 mask1001 experiment")
    for task in args.tasks:
        victim = Path(args.random_root)/f"{task}_LightGBM/best_pipeline.yaml"
        if victim_digest(victim) != CLEAN_RANDOM[task]:
            raise ValueError(f"Victim differs from original clean Random seed42: {victim}")
    output.mkdir(parents=True)
    marker = output/"INCOMPLETE.json"
    base.dump(marker, {"status":"preparing"})
    for task in args.tasks:
        with tempfile.TemporaryDirectory(dir=output, prefix="staging_") as temp:
            run_task(args,task,source,mcar,Path(temp),output)
    files = [f for f in output.rglob("*") if f.is_file() and f != marker]
    base.dump(output/"missingness_manifest.json", dict(version="savage_style_shared_mar_v1",
        repo=str(base.REPO), source=str(source), tasks=args.tasks, rate=.3, options=vars(args),
        lightgbm=lightgbm.__version__, optuna=optuna.__version__, mcar_manifest_sha256=base.sha(mcar/"missingness_manifest.json"),
        accepted_mcar_drift={rel: {"mcar": old, "current": base.sha(base.REPO/rel)}
                             for rel, old in manifest["code_hashes"].items()
                             if Path(rel).as_posix() in ACCEPTED_MCAR_DRIFT},
        code_hashes=base.code_fingerprints(),
        input_hashes={str(f.relative_to(output)):base.sha(f) for f in files}))
    verify_mcar(mcar)
    marker.unlink()
    try:
        base.verify(output)
    except Exception:
        base.dump(marker, {"status":"verification_failed"})
        raise


if __name__ == "__main__":
    main()
