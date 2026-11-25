"""Isolated missingness experiment. Never invokes the original dataset loader.

Run --help, or read docs/default_credit_missingness.md. Only new experiment
directories are written; existing experiments are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIELDS = {"A": ["LIMIT_BAL", "EDUCATION", "MARRIAGE"],
          "B": ["PAY_STATUS", "BILL_AMT", "PAY_AMT"]}
MONTHS = {"PAY_STATUS": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
          "BILL_AMT": [f"BILL_AMT{i}" for i in range(1, 7)],
          "PAY_AMT": [f"PAY_AMT{i}" for i in range(1, 7)]}
BASELINES = ["Random", "SAGA", "AlphaClean", "Learn2Clean", "AutoPrep", "DAGPipe", "DiffPrep"]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(f"Paths must be disjoint (no ancestor relationship): {a}, {b}")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def calibrate(z, rate):
    lo, hi = -40., 40.
    for _ in range(100):
        mid = (lo + hi) / 2
        if sigmoid(mid + 2 * z).mean() < rate:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def mechanism(layer, name, index):
    if name != "Mixed":
        return name
    return (["MNAR", "MAR", "MCAR"] if layer == "A"
            else ["MNAR", "MAR", "MCAR"])[index]


def inject(main, monthly, train_ids, layer, name, rate, seed):
    """Calibrate on fixed training IDs only. Common random numbers nest rates.

    B masks are user/field blocks covering all six months. All condition values
    are calculated from originals before masking. No label enters injection.
    """
    dirty_main, dirty_aux = main.copy(deep=True), monthly.copy(deep=True)
    mask_main = pd.DataFrame(False, index=main.index, columns=FIELDS["A"])
    mask_aux = pd.DataFrame(False, index=monthly.index, columns=FIELDS["B"])
    train = main.ID.isin(train_ids).to_numpy()
    audit = {}
    for i, col in enumerate(FIELDS[layer]):
        kind = mechanism(layer, name, i)
        values = (main[col] if layer == "A" else
                  main.ID.map(monthly.groupby("ID")[col].mean())).astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Original {col} has missing/nonfinite values; this protocol requires complete targets")
        condition = None
        if kind == "MCAR":
            z = np.zeros(len(main))
        elif kind == "MAR":
            condition = float(main.loc[train, "AGE"].median())
            z = np.where(main.AGE.to_numpy() < condition, 1., -1.)
        else:
            condition = float(np.median(values[train]))
            # Simulated low-limit/high-delay/high-bill/low-payment scenarios.
            higher = col in ("PAY_STATUS", "BILL_AMT", "EDUCATION", "MARRIAGE")
            z = np.where(values > condition if higher else values < condition, 1., -1.)
        alpha = calibrate(z[train], rate) if rate else -40.
        probability = sigmoid(alpha + 2 * z) if rate else np.zeros(len(main))
        # Stable field streams: identical masks across mechanisms when p matches;
        # within a mechanism lower-rate masks are subsets of higher-rate masks.
        rng = np.random.default_rng(np.random.SeedSequence([seed, ord(layer), i]))
        selected = rng.random(len(main)) < probability
        if layer == "A":
            if selected.any():
                dirty_main.loc[selected, col] = np.nan
            mask_main.loc[selected, col] = True
        else:
            selected_aux = monthly.ID.isin(main.loc[selected, "ID"])
            if selected_aux.any():
                dirty_aux.loc[selected_aux, col] = np.nan
            mask_aux.loc[selected_aux, col] = True
        audit[col] = {"mechanism": kind, "condition_threshold": condition,
                      "alpha": alpha, "beta": 2., "target_rate": rate,
                      "training_expected_rate": float(probability[train].mean()),
                      "selected_user_ids": main.loc[selected, "ID"].tolist()}
    return dirty_main, dirty_aux, mask_main, mask_aux, audit


# This loader is written ONLY into the new code snapshot. No monkey patch,
# source Excel read, pickle cache, download or reference to the original data.
LOADER = '''from pathlib import Path
import json
import pandas as pd
from ...dataset import TabularData

class DefaultCreditData(TabularData):
    def __init__(self, data_dir=None):
        super().__init__(name="DefaultCredit")
        if data_dir is None:
            raise ValueError("Isolated loader requires explicit scenario data_dir")
        self.data_dir = str(Path(data_dir).resolve())
        self.id_col, self.target_col = "ID", "TARGET"
        self.partitions = json.loads((Path(data_dir) / "partitions.json").read_text())

    def load_data(self):
        self.train_df = pd.read_parquet(Path(self.data_dir) / "main.parquet")
        self.test_df = None
        self.auxiliary_dfs["monthly_history"] = pd.read_parquet(Path(self.data_dir) / "monthly_history.parquet")
        return self.train_df, self.test_df

    def split(self, val_ratio=0.2, seed=42):
        prepared = self._prepared_splits()
        if prepared is not None:
            return prepared
        frame = self.train_df
        # Membership is frozen independently of model/search seed.
        result = {"test": self.test_df}
        for name in ("train", "val", "std_test"):
            part = frame.loc[frame.ID.isin(self.partitions[name])].drop(columns="__split__", errors="ignore").reset_index(drop=True)
            if name in ("train", "val") and part.empty:
                raise ValueError("Missing fixed partition: " + name)
            if len(part) or name != "std_test":
                result[name] = part
        return result
'''

SCREEN = '''import argparse, json, os
from pathlib import Path
import yaml
from baselines.common.evaluator import PipelineEvaluator
from baselines.common.pipeline import Pipeline

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=-1)
    p.add_argument("--methods", nargs="+", default=["median", "mean", "constant", "knn", "iterative"])
    a = p.parse_args()
    if a.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu_id)
    target = Path(a.output)
    if target.exists():
        raise FileExistsError(target)
    task = Path(__file__).resolve().parent / "dppbench/tasks/default_credit"
    # The runtime helper may live outside the immutable code snapshot.
    task = Path(os.environ.get("DPPBENCH_MISSINGNESS_CODE", str(task.parents[2]))) / "dppbench/tasks/default_credit"
    model_config = yaml.safe_load((task / "model.yaml").read_text())
    if a.gpu_id >= 0:
        model_config.setdefault("model_params", {}).update(device_type="cuda", gpu_device_id=0)
        print(f"LightGBM CUDA: physical GPU {a.gpu_id}, visible device 0", flush=True)
    rows = []
    for method in a.methods:
        config = yaml.safe_load((task / "prepare.yaml").read_text())
        for op in config["dag"]["ops"]:
            if op["op"] == "HandleMV":
                op["params"] = {"method": method}
                if method == "constant":
                    op["params"]["fill_value"] = 0
        evaluator = PipelineEvaluator(str(task), "default_credit", data_dir=a.data_dir, seed=a.seed,
                                      model_name="LightGBM", model_config=model_config)
        evaluator.set_evaluate_std_test(False)
        try:
            ev = evaluator.evaluate(Pipeline.from_yaml(yaml.safe_dump(config)))
            rows.append({"method": method, "seed": a.seed, "success": ev.success,
                         "fitness": ev.fitness if ev.success else None,
                         "error": ev.error, "metrics": ev.metrics,
                         "evaluation": "validation_only"})
            print(f"{method}: success={ev.success}, fitness={ev.fitness}, error={ev.error}", flush=True)
        finally:
            evaluator._executor.cleanup()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not all(r["success"] for r in rows):
        raise SystemExit("Some fixed methods failed; inspect results before selecting scenarios")
'''


def snapshot(output):
    destination = output / "code"
    destination.mkdir()
    hashes = {}
    for folder in ("scripts", "baselines", "dppbench"):
        for path in (REPO / folder).rglob("*"):
            relative = path.relative_to(REPO)
            if any(part.startswith(".") or part in {"data", "std_test", "ckpts", "__pycache__"}
                   for part in relative.parts):
                continue
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}:
                if path.is_symlink():
                    raise ValueError(f"Refusing source symlink: {path}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                hashes[str(relative)] = sha(path)
    for name in ("dataset.py",):
        shutil.copyfile(REPO / name, destination / name)
        hashes[name] = sha(REPO / name)
    loader = destination / "dppbench/tasks/default_credit/default_credit_data.py"
    loader.write_text(LOADER, encoding="utf-8")
    # prev-only relation DAG binds its secondary source automatically. The
    # original YAML also names aux_df, which this executor rejects as duplicate.
    # Normalize the isolated copy only; never edit the running task's YAML.
    import yaml
    prepare_path = destination / "dppbench/tasks/default_credit/prepare.yaml"
    config = yaml.safe_load(prepare_path.read_text(encoding="utf-8"))
    for op in config["dag"]["ops"]:
        if op["op"] == "JoinTable" and len(op.get("prev", [])) > 1:
            op.get("params", {}).pop("aux_df", None)
    prepare_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (destination / "screen_missingness.py").write_text(SCREEN, encoding="utf-8")
    dump(output / "source_code_hashes.json", hashes)


def prepare(args):
    from sklearn.model_selection import train_test_split
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    disjoint(output, source)
    disjoint(output, REPO)
    if output.exists():
        raise FileExistsError(f"Use a NEW experiment directory: {output}")
    task = source / "default_credit"
    paths = [task / "std_test" / name for name in
             ("train_frozen.parquet", "std_test.parquet", "meta.json")]
    raw = (Path(args.wide_parquet).resolve() if args.wide_parquet else
           task / "data/default of credit card clients.xls")
    disjoint(output, raw)
    for path in [*paths, raw]:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path): sha(path) for path in [*paths, raw]}
    pool, test = (pd.read_parquet(p).drop(columns="__split__", errors="ignore") for p in paths[:2])
    for frame in (pool, test):
        if not frame.ID.is_unique or frame.ID.isna().any() or frame.TARGET.isna().any():
            raise ValueError("Frozen IDs must be unique and labels nonmissing")
    if set(pool.ID) & set(test.ID):
        raise ValueError("Frozen train/test ID overlap")
    train, val = train_test_split(pool, test_size=0.2, random_state=42, stratify=pool.TARGET)
    partitions = {"train": train.ID.tolist(), "val": val.ID.tolist(), "std_test": test.ID.tolist()}
    main = pd.concat([pool, test], ignore_index=True)
    wide = pd.read_parquet(raw) if args.wide_parquet else pd.read_excel(raw, header=1)
    wide = wide.rename(columns={"default payment next month": "TARGET"})
    if not wide.ID.is_unique or not set(main.ID) <= set(wide.ID):
        raise ValueError("Raw IDs do not match frozen data")
    wide = wide.set_index("ID").loc[main.ID].reset_index()
    if not np.array_equal(wide.TARGET.to_numpy(), main.TARGET.to_numpy()):
        raise ValueError("Raw labels disagree with frozen data")
    aux = pd.concat([pd.DataFrame({"ID": wide.ID, "MONTH_OFFSET": m,
                                  **{f: wide[cols[m]] for f, cols in MONTHS.items()}})
                     for m in range(6)], ignore_index=True)
    if main[FIELDS["A"] + ["SEX", "AGE"]].isna().any().any():
        raise ValueError("Original main features must be complete")
    if not np.isfinite(aux[FIELDS["B"]].to_numpy(dtype=float)).all():
        raise ValueError("Original history must be complete and finite")
    output.mkdir(parents=True)
    dump(output / "INCOMPLETE.json", {"status": "preparing; do not run"})
    snapshot(output)
    original = output / "originals"
    original.mkdir()
    main.to_parquet(original / "main.parquet", index=False)
    aux.to_parquet(original / "monthly_history.parquet", index=False)
    dump(original / "partitions.json", partitions)
    specs = [("clean", "A", "MCAR", 0., 0)]
    specs += [(f"{layer}_{kind}_q{round(rate*100):02d}_mask{seed}", layer, kind, rate, seed)
              for seed in args.mask_seeds for layer in args.layers
              for kind in args.mechanisms for rate in args.rates]
    if len({s[0] for s in specs}) != len(specs):
        raise ValueError("Duplicate scenario IDs")
    scenarios = []
    for name, layer, kind, rate, seed in specs:
        folder = output / "scenarios" / name
        datadir = folder / "datasets/default_credit/data"
        std = folder / "datasets/default_credit/std_test"
        datadir.mkdir(parents=True)
        std.mkdir()
        dirty, dirty_aux, mask, mask_aux, audit = inject(main, aux, partitions["train"], layer, kind, rate, seed)
        dirty.to_parquet(datadir / "main.parquet", index=False)
        dirty_aux.to_parquet(datadir / "monthly_history.parquet", index=False)
        dump(datadir / "partitions.json", partitions)
        dirty[dirty.ID.isin(pool.ID)].to_parquet(std / "train_frozen.parquet", index=False)
        dirty[dirty.ID.isin(test.ID)].to_parquet(std / "std_test.parquet", index=False)
        shutil.copyfile(paths[2], std / "meta.json")
        mask.assign(ID=main.ID).to_parquet(folder / "mask_main.parquet", index=False)
        mask_aux.assign(ID=aux.ID, MONTH_OFFSET=aux.MONTH_OFFSET).to_parquet(folder / "mask_history.parquet", index=False)
        for col, info in audit.items():
            info["realized_user_rates"] = {
                split: float(main.loc[main.ID.isin(ids), "ID"].isin(info["selected_user_ids"]).mean())
                for split, ids in partitions.items()}
            # Demographic rates support audit without asserting real-world bias.
            info["sex_rates"] = {str(group): float(part.ID.isin(info["selected_user_ids"]).mean())
                                 for group, part in main.groupby("SEX")}
        metadata = {"name": name, "layer": layer, "mechanism": kind,
                    "rate": rate, "mask_seed": seed, "conditions": audit,
                    "protocol": "same_mechanism_same_target_rate_train_val_test",
                    "partition_seed": 42, "validation_ratio": 0.2,
                    "originals": "../../originals", "test_used_for_calibration": False}
        dump(folder / "manifest.json", metadata)
        scenarios.append(name)
    after = {path: sha(path) for path in before}
    if before != after:
        raise RuntimeError("Source changed during preparation; do not use experiment")
    dump(output / "source_data_hashes.json", before)
    dump(output / "suite.json", {"protocol_version": 1, "scenarios": scenarios,
                                "source_root": str(source), "source_repo": str(REPO),
                                "note": "New code/data copies only. No original loader/cache invoked."})
    dump(output / "integrity.json", {
        str(p.relative_to(output)): sha(p)
        for folder in ("code", "originals", "scenarios")
        for p in (output / folder).rglob("*") if p.is_file()
    })
    (output / "INCOMPLETE.json").unlink()
    print(f"Prepared {len(scenarios)} isolated scenarios: {output}")


def verify(root):
    root = Path(root).resolve()
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Preparation incomplete")
    hashes = json.loads((root / "integrity.json").read_text())
    originals = json.loads((root / "source_data_hashes.json").read_text())
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Experiment input changed: {path}")
    for path, expected in originals.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    print("Verified original source and experiment input hashes", flush=True)


def reserve_result(result, root, retry_failed=False):
    result, root = Path(result), Path(root).resolve()
    result.resolve().relative_to(root)
    if result.exists() and retry_failed:
        if result.is_symlink():
            raise ValueError(f"Refusing result symlink: {result}")
        status_path = result / "run_status.json"
        validation = result / "validation.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        records = json.loads(validation.read_text()) if validation.exists() else []
        if status.get("status") == "complete" or (records and all(r.get("success") for r in records)):
            raise FileExistsError(f"Successful output will not be overwritten: {result}")
        if status.get("status") == "running":
            raise ValueError(f"Output may still be running: {result}")
        if status.get("status") != "failed" and result.name != "fixed":
            raise ValueError(f"No confirmed failure status; refusing retry: {result}")
        archive = result.with_name(result.name + "_failed_" + uuid.uuid4().hex[:12])
        archive.resolve().relative_to(root)
        result.rename(archive)
        print(f"Preserved previous attempt: {archive}", flush=True)
    result.mkdir(parents=True, exist_ok=False)


def run(args):
    root = Path(args.output_root).resolve()
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Preparation incomplete")
    suite = json.loads((root / "suite.json").read_text())
    disjoint(root, suite["source_root"])
    disjoint(root, suite["source_repo"])
    verify(root)
    selected = args.scenarios or suite["scenarios"]
    if not set(selected) <= set(suite["scenarios"]):
        raise ValueError("Unknown scenario name")
    if args.command == "baselines" and args.execute:
        if not args.scenarios:
            raise ValueError("Select validated scenarios explicitly with --scenarios before final testing")
        for scenario in set(selected) | {"clean"}:
            for seed in args.search_seeds:
                validation = root / "results/screen" / scenario / f"seed{seed}/fixed/validation.json"
                if not validation.is_file():
                    raise ValueError(f"Run fixed-method screening first: {validation}")
                records = json.loads(validation.read_text())
                if not records or not all(r["success"] for r in records):
                    raise ValueError(f"Resolve screening failures first: {validation}")
    code = root / "code"
    for scenario in selected:
        data = root / "scenarios" / scenario / "datasets"
        for seed in args.search_seeds:
            labels = args.baselines if args.command == "baselines" else ["fixed"]
            for label in labels:
                result = root / "results" / args.command / scenario / f"seed{seed}" / label
                command = [sys.executable]
                if args.command == "screen":
                    # Versioned helper outside code/ supports existing prepared
                    # suites without changing their snapshots or integrity hashes.
                    helper = root / "runtime" / ("screen_" + hashlib.sha256(SCREEN.encode()).hexdigest()[:16] + ".py")
                    if args.execute:
                        helper.parent.mkdir(exist_ok=True)
                        if helper.exists():
                            if helper.read_text(encoding="utf-8") != SCREEN:
                                raise ValueError(f"Runtime helper changed: {helper}")
                        else:
                            with helper.open("x", encoding="utf-8") as stream:
                                stream.write(SCREEN)
                    command += [str(helper), "--data-dir", str(data),
                                "--output", str(result / "validation.json"), "--seed", str(seed),
                                "--gpu-id", str(args.gpu_id),
                                "--methods", *args.methods]
                else:
                    command += [str(code / "scripts/evaluate_baseline.py"), "--baseline", label,
                                "--data_names", "default_credit", "--data_dir", str(data),
                                "--output_dir", str(result), "--seed", str(seed), "--skip_build",
                                "--gpu_id", str(args.gpu_id), "--max_pipeline_len", "10",
                                "--max_evaluations", str(args.max_evaluations)]
                    for option, value in (("--max_memory_mb", args.max_memory_mb),
                                          ("--max_gpu_memory_mb", args.max_gpu_memory_mb)):
                        if value is not None:
                            command += [option, str(value)]
                    if args.final_hpo:
                        command += ["--final_hpo"]
                print(subprocess.list2cmdline(command), flush=True)
                if args.execute:
                    # Do not silently reuse partial/successful outputs.
                    reserve_result(result, root, args.retry_failed)
                    env = dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE="1")
                    env["PYTHONUNBUFFERED"] = "1"
                    env["DPPBENCH_MISSINGNESS_CODE"] = str(code)
                    env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(root / "tmp")
                    (root / "tmp").mkdir(exist_ok=True)
                    dump(result / "run_status.json", {"status": "running", "command": command})
                    try:
                        with (result / "run.log").open("w", encoding="utf-8") as log:
                            if args.tee_log:
                                with subprocess.Popen(command, cwd=code, env=env,
                                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                      text=True, encoding="utf-8", errors="replace") as process:
                                    for line in process.stdout:
                                        log.write(line)
                                        log.flush()
                                        print(line, end="", flush=True)
                                    returncode = process.wait()
                                if returncode:
                                    raise subprocess.CalledProcessError(returncode, command)
                            else:
                                subprocess.run(command, cwd=code, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, check=True)
                    except subprocess.CalledProcessError as exc:
                        dump(result / "run_status.json", {"status": "failed", "returncode": exc.returncode})
                        tail = "\n".join((result / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                        print(f"\nChild process failed. Log: {result / 'run.log'}\n{tail}", file=sys.stderr, flush=True)
                        raise SystemExit(exc.returncode)
                    dump(result / "run_status.json", {"status": "complete"})
    if args.execute:
        verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create new isolated code/data copies, never overwrite")
    p.add_argument("--source-root", required=True, help="Existing dataset root containing default_credit/")
    p.add_argument("--output-root", required=True, help="NEW directory disjoint from source data AND repository")
    p.add_argument("--wide-parquet", help="Optional original wide table parquet instead of reading XLS")
    p.add_argument("--mask-seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    p.add_argument("--layers", nargs="+", choices=list(FIELDS), default=list(FIELDS))
    p.add_argument("--mechanisms", nargs="+", choices=["MCAR", "MAR", "MNAR", "Mixed"], default=["MCAR", "MAR", "MNAR", "Mixed"])
    p.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    for name in ("screen", "baselines"):
        p = sub.add_parser(name, help="Print commands; add --execute to actually run sequentially")
        p.add_argument("--output-root", required=True)
        p.add_argument("--scenarios", nargs="+")
        p.add_argument("--search-seeds", nargs="+", type=int, default=[42, 43, 44])
        p.add_argument("--execute", action="store_true")
        p.add_argument("--tee-log", action="store_true", help="Stream complete child stdout/stderr to console as well as run.log")
        p.add_argument("--retry-failed", action="store_true", help="Archive failed output before retrying; never overwrite successful output")
        if name == "screen":
            p.add_argument("--gpu-id", type=int, default=-1, help="Physical GPU index for CUDA-enabled LightGBM; imputers remain on CPU")
            p.add_argument("--methods", nargs="+", choices=["mean", "median", "constant", "knn", "iterative"], default=["mean", "median", "constant", "knn", "iterative"])
        else:
            p.add_argument("--baselines", nargs="+", choices=BASELINES, default=BASELINES)
            p.add_argument("--gpu-id", type=int, default=-1)
            p.add_argument("--max-evaluations", type=int, default=1000, help="Ceiling only; native search counts differ")
            p.add_argument("--max-memory-mb", type=float, default=None)
            p.add_argument("--max-gpu-memory-mb", type=float, default=None)
            p.add_argument("--final-hpo", action="store_true")
    p = sub.add_parser("verify", help="Read-only hash check of original and experiment inputs")
    p.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if any(not 0 < r < 1 for r in args.rates):
            parser.error("rates must be between 0 and 1")
        if any(s < 0 for s in args.mask_seeds):
            parser.error("mask seeds must be nonnegative")
        prepare(args)
    elif args.command == "verify":
        verify(args.output_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
