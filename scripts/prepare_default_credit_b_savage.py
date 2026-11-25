"""Fixed clean Random victim, B-layer user/history blocks, matched MCAR counts."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_default_credit_b_original_protocol as credit
import prepare_multidata_savage as attack

TASK = "default_credit"
COLS = list(credit.MONTHS)
CONFIG = {"cols": COLS, "target": "TARGET",
          "class": "dppbench.tasks.default_credit.default_credit_data.DefaultCreditData"}
DIGEST = "8aba5b2cf5da653154473d3fa62f29da7277b05ee9828437672f880ce63b588e"


def prepare(args):
    mcar, output = Path(args.mcar_root).resolve(), Path(args.output_root).resolve()
    pipeline = Path(args.pipeline).resolve()
    credit.disjoint(mcar, output)
    credit.disjoint(pipeline, output)
    if attack.victim_digest(pipeline) != DIGEST:
        raise ValueError("Expected original clean seed42 Random pipeline for default_credit")
    credit.verify(mcar)
    protocol = json.loads((mcar/"protocol.json").read_text())
    if (protocol["mechanism"], protocol["rate"], protocol["mask_seed"],
        Path(protocol["source_root"]).resolve(), protocol["original_run_seeds"]) != (
            "MCAR", .3, 1001, Path(args.source_root).resolve(), [42,43,44]):
        raise ValueError("Expected matching B MCAR30 mask1001 reference")

    def select(main, monthly, train_ids, layer, name, rate, seed):
        # Encode one candidate deletion per user/field. These marker columns
        # never enter the victim pipeline or downstream model.
        users = main.assign(**{c: 1. for c in COLS})
        ref = pd.DataFrame({c: main.ID.isin(protocol["conditions"][c]["selected_user_ids"])
                            for c in COLS})
        cache = pd.read_pickle(mcar/TASK/"data"/(credit.XLS+".pkl"))
        expected = monthly.copy()
        for c in COLS:
            expected.loc[expected.ID.isin(main.ID[ref[c]]), c] = np.nan
        pd.testing.assert_frame_equal(expected, credit.history(cache))
        pool = pd.read_parquet(mcar/TASK/"std_test/train_frozen.parquet")
        id_to_row = pd.Series(np.arange(len(main)), index=main.ID)
        pool_rows = id_to_row.loc[pool.ID].to_numpy()
        outer, parts = attack.inner_data(pool, TASK, args.max_attack_rows)
        rows = pool_rows[outer]
        frame = users.iloc[rows].reset_index(drop=True)
        ref_dev = ref.iloc[rows].reset_index(drop=True)
        aux = monthly[monthly.ID.isin(frame.ID)]
        victim = attack.make_victim(TASK, mcar, folder/"random_seed42.yaml")
        make_data = victim._make_configured_data_instance
        def evaluate(candidate):
            history = aux.copy()
            for c in COLS:
                history.loc[history.ID.isin(candidate.loc[candidate[c].isna(), "ID"]), c] = np.nan
            def instance():
                data = make_data()
                data.auxiliary_dfs["monthly_history"] = history.copy()
                return data
            with patch.object(victim, "_make_configured_data_instance", instance):
                return attack.score(victim, candidate[main.columns])
        try:
            clean = evaluate(frame)
            control = evaluate(attack.apply_mask(frame, ref_dev))
            counts = [ref_dev.iloc[idx].sum().to_dict() for idx in parts]
            rule, auc = attack.search(TASK, frame, parts, counts, evaluate, args, folder)
        finally:
            victim.cleanup()
        credit.dump(folder/"selected.json", dict(rule=rule, clean_dev_auc=clean, mcar_dev_auc=control,
                    selected_dev_auc=auc, fit_ids=frame.ID.iloc[parts[0]].tolist(),
                    dev_ids=frame.ID.iloc[parts[1]].tolist(), victim_digest=DIGEST))
        partitions = [pool_rows, np.flatnonzero(~main.ID.isin(pool.ID))]
        budgets = [ref.iloc[idx].sum().to_dict() for idx in partitions]
        mask = attack.exact_mask(users, COLS, rule, partitions, budgets, seed, TASK)
        dirty = monthly.copy()
        history_mask = pd.DataFrame(index=monthly.index)
        audit = {}
        for c in COLS:
            ids = main.loc[mask[c], "ID"].tolist()
            history_mask[c] = monthly.ID.isin(ids)
            dirty.loc[history_mask[c], c] = np.nan
            audit[c] = dict(mechanism="SAVAGE", rule=rule, selected_user_ids=ids)
        return main.copy(), dirty, None, history_mask, audit

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        shutil.copyfile(pipeline, folder/"random_seed42.yaml")
        if attack.victim_digest(folder/"random_seed42.yaml") != DIGEST:
            raise ValueError("Copied pipeline fingerprint differs")
        with patch.object(credit, "inject", select), patch.dict(attack.base.TASKS, {TASK: CONFIG}), patch.dict(
                attack.FIELDS, {TASK: ["AGE", "LIMIT_BAL", "EDUCATION"]}):
            credit.prepare(args)
        shutil.copytree(folder, output/"attack"/TASK)
    metadata = json.loads((output/"protocol.json").read_text())
    metadata["calibration"] = "inner_fit_from_seed42_training_pool"
    selected = json.loads((output/"attack"/TASK/"selected.json").read_text())
    metadata["calibration_ids"] = selected["fit_ids"]
    credit.dump(output/"protocol.json", metadata)
    credit.verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root","mcar-root","pipeline","output-root"):
        parser.add_argument("--"+flag, required=True)
    parser.set_defaults(rate=.3, mask_seed=1001, run_seeds=[42,43,44], mechanism="SAVAGE",
                        max_attack_rows=12000, beam=2, trials=6)
    prepare(parser.parse_args())
