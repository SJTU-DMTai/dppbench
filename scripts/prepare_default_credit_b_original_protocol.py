"""Data-only B-layer perturbation for the ORIGINAL exp1 evaluator and loader.

No replacement loader, code snapshot, split override, or configuration edit.
Copies source XLS/frozen artifacts to a NEW root and supplies the original
Excel cache format with only selected historical values masked. The source
XLS copy is a provenance backing file; the verified cache is the actual input.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from default_credit_missingness import REPO, MONTHS, inject, sha, dump, disjoint

XLS = "default of credit card clients.xls"


def history(wide):
    # Exact original loader order: ID, PAY_STATUS, BILL_AMT, PAY_AMT,
    # MONTH_OFFSET; original XLS row order within each month.
    records = []
    for month in range(6):
        cols = [MONTHS[field][month] for field in MONTHS]
        part = wide[["ID", *cols]].rename(columns=dict(zip(cols, MONTHS)))
        part["MONTH_OFFSET"] = month
        records.append(part)
    return pd.concat(records, ignore_index=True)


def check_values(original, dirty, allowed):
    """Only finite/originally observed values may become missing."""
    if list(original.columns) != list(dirty.columns) or not original.index.equals(dirty.index):
        raise ValueError("Column or row order changed")
    for col in original:
        if col not in allowed:
            pd.testing.assert_series_equal(original[col], dirty[col])
        else:
            changed = ~((original[col] == dirty[col]) | (original[col].isna() & dirty[col].isna()))
            if not (original.loc[changed, col].notna() & dirty.loc[changed, col].isna()).all():
                raise ValueError(f"Unexpected non-mask change: {col}")


def prepare(args):
    from sklearn.model_selection import train_test_split
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    disjoint(source, output)
    disjoint(REPO, output)
    if output.exists():
        raise FileExistsError(f"Use a new root: {output}")
    task = source / "default_credit"
    xls = task / "data" / XLS
    cache = Path(str(xls) + ".pkl")
    frozen = task / "std_test"
    inputs = [xls, *(frozen / f for f in ("train_frozen.parquet", "std_test.parquet", "meta.json"))]
    if cache.exists():
        inputs.append(cache)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path): sha(path) for path in inputs}
    # Follow the original cache selection exactly, without calling the helper
    # that could write a cache into the live source directory.
    if cache.exists() and cache.stat().st_mtime >= xls.stat().st_mtime:
        wide = pd.read_pickle(cache)
        read_method = "existing_original_pickle_cache"
    else:
        wide = pd.read_excel(xls, header=1)
        read_method = "original_excel_header1"
    original_wide = wide.copy(deep=True)
    main = wide.drop(columns=[col for cols in MONTHS.values() for col in cols]).rename(
        columns={"default payment next month": "TARGET"}).reset_index(drop=True)
    aux = history(wide)
    pool = pd.read_parquet(frozen / "train_frozen.parquet")
    test = pd.read_parquet(frozen / "std_test.parquet")
    if not main.ID.is_unique or not pool.ID.is_unique or not test.ID.is_unique:
        raise ValueError("IDs must be unique")
    if set(pool.ID) & set(test.ID) or set(pool.ID) | set(test.ID) != set(main.ID):
        raise ValueError("Frozen train/test membership does not match raw IDs")
    # Verify frozen feature values correspond to this original loader input.
    for frame in (pool, test):
        expected = main.set_index("ID").loc[frame.ID].reset_index()
        actual = frame.drop(columns="__split__", errors="ignore").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected[actual.columns], actual, check_dtype=False)

    # Reproduce each ORIGINAL split, but do not use it to override evaluation.
    # The common mask is calibrated only on IDs in TRAIN for every requested
    # run seed, so no run's validation records are used for calibration.
    train_sets, split_audit = [], {}
    for seed in args.run_seeds:
        train, val = train_test_split(pool, test_size=.2, random_state=seed, stratify=pool.TARGET)
        train_sets.append(set(train.ID))
        split_audit[str(seed)] = {"train_ids": train.ID.tolist(), "val_ids": val.ID.tolist()}
    calibration_ids = set.intersection(*train_sets)
    if not calibration_ids:
        raise ValueError("Empty training intersection for missingness calibration")
    dirty_main, dirty_aux, _, aux_mask, audit = inject(
        main, aux, calibration_ids, "B", args.mechanism, args.rate, args.mask_seed)
    pd.testing.assert_frame_equal(dirty_main, main)
    for field, info in audit.items():
        chosen = wide.ID.isin(info["selected_user_ids"])
        for column in MONTHS[field]:
            if chosen.any():
                wide.loc[chosen, column] = np.nan
        info["observed_user_rates"] = {
            "frozen_pool": float(pool.ID.isin(info["selected_user_ids"]).mean()),
            "std_test": float(test.ID.isin(info["selected_user_ids"]).mean())}
    allowed = [col for cols in MONTHS.values() for col in cols]
    check_values(original_wide, wide, allowed)
    pd.testing.assert_frame_equal(history(wide), dirty_aux)

    output.mkdir(parents=True)
    dump(output / "INCOMPLETE.json", {"status": "preparing"})
    data_out, std_out = output / "default_credit/data", output / "default_credit/std_test"
    data_out.mkdir(parents=True)
    std_out.mkdir()
    shutil.copy2(xls, data_out / XLS)
    target_cache = Path(str(data_out / XLS) + ".pkl")
    wide.to_pickle(target_cache)
    # Make cache selection robust to a copied XLS with a future mtime.
    stamp = max(target_cache.stat().st_mtime, (data_out / XLS).stat().st_mtime + 1)
    os.utime(target_cache, (stamp, stamp))
    for filename in ("train_frozen.parquet", "std_test.parquet", "meta.json"):
        shutil.copy2(frozen / filename, std_out / filename)
    # B changes ONLY auxiliary history, so frozen main tables are byte copies.
    for path in std_out.iterdir():
        if sha(path) != sha(frozen / path.name):
            raise ValueError(f"Frozen artifact changed: {path}")
    originals = output / "audit"
    originals.mkdir()
    original_wide.to_parquet(originals / "original_wide.parquet", index=False)
    aux_mask.assign(ID=aux.ID, MONTH_OFFSET=aux.MONTH_OFFSET).to_parquet(
        originals / "mask_history.parquet", index=False)
    dump(originals / "original_split_memberships.json", split_audit)
    dump(originals / "source_data_hashes.json", before)
    # Fingerprint the original implementation/configs without modifying them.
    code_hashes = {}
    for folder in ("scripts", "baselines", "dppbench"):
        for path in (REPO / folder).rglob("*"):
            rel = path.relative_to(REPO)
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml") and not any(
                p.startswith(".") or p in ("data", "std_test", "__pycache__", "ckpts") for p in rel.parts):
                code_hashes[str(rel)] = sha(path)
    dump(originals / "original_code_hashes.json", code_hashes)
    dump(output / "protocol.json", {
        "version": "original_exp1_data_only_B_v1", "source_root": str(source),
        "original_repo": str(REPO), "mechanism": args.mechanism, "rate": args.rate,
        "mask_seed": args.mask_seed, "original_run_seeds": args.run_seeds,
        "loader": "original_DefaultCreditData_unchanged", "read_method": read_method,
        "split_override": False, "frozen_main_artifacts": "byte_identical_to_source",
        "calibration": "intersection_of_original_training_ids_across_run_seeds",
        "calibration_ids": sorted(calibration_ids), "conditions": audit,
        "note": "Only --data_dir and output paths change in original exp1 invocation. The original loader consumes the masked XLS-format pickle cache; do not delete or invalidate it."})
    # Exercise the unchanged original loader on the copy and original splitting.
    sys.path.insert(0, str(REPO))
    from dppbench.tasks.default_credit.default_credit_data import DefaultCreditData
    loaded = DefaultCreditData(str(data_out))
    loaded.load_data()
    pd.testing.assert_frame_equal(loaded.train_df, main)
    pd.testing.assert_frame_equal(loaded.auxiliary_dfs["monthly_history"], dirty_aux)
    loaded.load_std_test_frozen()
    for seed in args.run_seeds:
        parts = loaded.split(seed=seed)
        if parts["train"].ID.tolist() != split_audit[str(seed)]["train_ids"] or parts["val"].ID.tolist() != split_audit[str(seed)]["val_ids"]:
            raise ValueError(f"Original split/order mismatch for seed {seed}")
    if before != {path: sha(path) for path in before}:
        raise RuntimeError("Original source changed during preparation")
    dump(output / "input_hashes.json", {str(p.relative_to(output)): sha(p)
        for folder in (data_out, std_out) for p in folder.iterdir() if p.is_file()})
    (output / "INCOMPLETE.json").unlink()
    print(f"Prepared data-only original-protocol B perturbation: {output}")
    print("Original loader and per-seed split membership/order verified; frozen main files byte-identical.")


def verify(root):
    root = Path(root).resolve()
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Preparation incomplete")
    for rel, digest in json.loads((root / "input_hashes.json").read_text()).items():
        if sha(root / rel) != digest:
            raise ValueError(f"Perturbed input changed: {rel}")
    for path, digest in json.loads((root / "audit/source_data_hashes.json").read_text()).items():
        if sha(path) != digest:
            raise ValueError(f"Original source changed: {path}")
    protocol = json.loads((root / "protocol.json").read_text())
    repo = Path(protocol["original_repo"])
    for rel, digest in json.loads((root / "audit/original_code_hashes.json").read_text()).items():
        if sha(repo / rel) != digest:
            raise ValueError(f"Original code/config changed after preparation: {rel}")
    print("Original source and data-only perturbation input hashes verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mechanism", choices=["MCAR", "MAR", "MNAR", "Mixed"], default="Mixed")
    parser.add_argument("--rate", type=float, default=.3)
    parser.add_argument("--mask-seed", type=int, default=1001)
    parser.add_argument("--run-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output_root)
    else:
        if not args.source_root or not 0 <= args.rate < 1 or args.mask_seed < 0:
            parser.error("Require source-root, 0 <= rate < 1, mask-seed >= 0")
        prepare(args)


if __name__ == "__main__":
    main()
