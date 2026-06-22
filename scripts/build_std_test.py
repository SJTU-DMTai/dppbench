"""Generate the per-task standard test set ("std-test") used by every
baseline harness for fair, apples-to-apples evaluation.

For each task this script:
  * Calls ``data.load_data()`` to obtain the task's canonical DataFrames:
    raw data plus any fixed dataset-level protocol such as Amazon Beauty's
    k-core. The std-test stays free of pipeline-specific feature engineering
    decisions.
  * Holds out a fixed slice of rows by a task-appropriate rule:
      - tabular binary classification: stratified random 20% (seed=42).
      - tabular time-series regression: chronological tail 20%.
      - tabular graph (elliptic_bitcoin): node-id random 20% (seed=42).
      - rec leave-one-out: each user's last interaction (by timestamp,
        ties broken by row order) goes to std-test; we additionally
        sample a fixed set of 100 negative items per std-test positive
        with seed=42 so that ranking metrics are stable across baselines.
  * Writes raw artefacts to ``dppbench/tasks/<task>/std_test/``:
      - ``std_test.parquet``          (the held-out rows, with labels)
      - ``train_frozen.parquet``      (tabular only — the rest of train)
      - ``interaction_frozen.parquet`` (rec only — interactions w/o std-test)
      - ``std_test_negatives.parquet`` (rec only — fixed negatives)
      - ``meta.json``

Run once::

    python scripts/build_std_test.py

Re-running is idempotent: the seed and rules are fixed, so produced
files are byte-identical (modulo parquet metadata).
"""

import argparse
import copy
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Task registry: (task_name, dotted_class, kind)
# kind ∈ {"binary", "timeseries", "graph", "rec_loo"}.
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    # --- tabular binary classification ---------------------------------
    "berka": ("dppbench.tasks.berka.berka_data.BerkaData", "binary"),
    "bondora": ("dppbench.tasks.bondora.bondora_data.BondoraData", "binary"),
    "default_credit": (
        "dppbench.tasks.default_credit.default_credit_data.DefaultCreditData",
        "binary",
    ),
    "fraud_detection": (
        "dppbench.tasks.fraud_detection.fraud_detection_data.FraudDetectionData",
        "binary",
    ),
    "home_credit": (
        "dppbench.tasks.home_credit.home_credit_data.HomeCreditData",
        "binary",
    ),
    "polish_bankruptcy": (
        "dppbench.tasks.polish_bankruptcy.polish_bankruptcy_data.PolishBankruptcyData",
        "binary",
    ),
    # --- tabular time-series regression --------------------------------
    "beijing_air_quality": (
        "dppbench.tasks.beijing_air_quality.beijing_air_quality_data.BeijingAirQualityData",
        "timeseries",
    ),
    "bike_sharing": (
        "dppbench.tasks.bike_sharing.bike_sharing_data.BikeSharingData",
        "timeseries",
    ),
    "citibike_jc_hourly": (
        "dppbench.tasks.citibike_jc_hourly.citibike_jc_hourly_data.CitibikeJcHourlyData",
        "timeseries",
    ),
    "nyc_taxi_hourly": (
        "dppbench.tasks.nyc_taxi_hourly.nyc_taxi_hourly_data.NycTaxiHourlyData",
        "timeseries",
    ),
    # --- tabular graph -------------------------------------------------
    "elliptic_bitcoin": (
        "dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data.EllipticBitcoinData",
        "graph",
    ),
    # --- rec ----------------------------------------------------------
    "amazon_beauty": (
        "dppbench.tasks.amazon_beauty.amazon_beauty_data.AmazonBeautyData",
        "rec_loo",
    ),
    "kuairec": ("dppbench.tasks.kuairec.kuairec_data.KuairecData", "rec_loo"),
    "movielens": (
        "dppbench.tasks.movielens.movielens_data.MovielensData",
        "rec_loo",
    ),
    "yelp": ("dppbench.tasks.yelp.yelp_data.YelpData", "rec_loo"),
    "tenrec": ("dppbench.tasks.tenrec.tenrec_data.TenrecData", "rec_loo"),
}

STD_TEST_SEED = 42
STD_TEST_FRAC = 0.20
REC_NUM_NEGATIVES = 100

# Time-series tasks expose a numeric chronological column on
# ``self._sort_col`` after ``load_data()`` (except citibike & nyc_taxi
# which create that column only inside ResampleTimeSeries). For those,
# fall back to a raw timestamp column.
TIMESERIES_FALLBACK_TIME_COL = {
    "citibike_jc_hourly": "started_at",
    "nyc_taxi_hourly": "tpep_pickup_datetime",
}


def _import_class(dotted):
    module_path, class_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _task_data_dir(task_name, data_dir=None):
    if not data_dir:
        return None
    return os.path.join(os.path.abspath(data_dir), task_name, "data")


def _std_test_dir(task_name, data_dir=None):
    task_data_dir = _task_data_dir(task_name, data_dir)
    if task_data_dir:
        base = os.path.dirname(task_data_dir)
    else:
        base = os.path.join(
            os.path.dirname(__file__), "..", "dppbench", "tasks", task_name
        )
    out = os.path.join(base, "std_test")
    os.makedirs(out, exist_ok=True)
    return out


def _model_yaml_path(task_name):
    return os.path.join(
        os.path.dirname(__file__), "..", "dppbench", "tasks", task_name, "model.yaml"
    )


def _load_model_config(task_name):
    path = _model_yaml_path(task_name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def canonical_data_protocol(task_name):
    if task_name not in TASK_REGISTRY:
        return None
    dotted, _ = TASK_REGISTRY[task_name]
    data_cls = _import_class(dotted)
    protocol = getattr(data_cls, "CANONICAL_DATA_PROTOCOL", None)
    return copy.deepcopy(protocol) if protocol else None


def _save_meta(out_dir, meta):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Tabular binary classification
# ---------------------------------------------------------------------------
def build_binary(task_name, data, out_dir, dry_run):
    from sklearn.model_selection import train_test_split

    df = data.train_df
    target = data.target_col
    if target not in df.columns:
        raise ValueError(f"{task_name}: target_col '{target}' not in train_df")

    y = df[target]
    train_idx, std_idx = train_test_split(
        np.arange(len(df)),
        test_size=STD_TEST_FRAC,
        random_state=STD_TEST_SEED,
        stratify=y,
    )
    train_frozen = df.iloc[train_idx].reset_index(drop=True)
    std_test = df.iloc[std_idx].reset_index(drop=True)

    meta = {
        "task": task_name,
        "kind": "binary",
        "split_method": "stratified_holdout",
        "seed": STD_TEST_SEED,
        "test_size": STD_TEST_FRAC,
        "train_frozen_rows": len(train_frozen),
        "std_test_rows": len(std_test),
        "target_col": target,
        "positive_ratio_train": float(y.iloc[train_idx].mean()),
        "positive_ratio_std_test": float(std_test[target].mean()),
    }
    if dry_run:
        return meta

    train_frozen.to_parquet(os.path.join(out_dir, "train_frozen.parquet"))
    std_test.to_parquet(os.path.join(out_dir, "std_test.parquet"))
    _save_meta(out_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# Tabular time-series regression: chronological tail 20% of unique timestamps
# ---------------------------------------------------------------------------
def build_timeseries(task_name, data, out_dir, dry_run):
    df = data.train_df.copy()

    # Resolve the ordering column.
    sort_col = getattr(data, "_sort_col", None)
    if sort_col is None or sort_col not in df.columns:
        sort_col = TIMESERIES_FALLBACK_TIME_COL.get(task_name)
        if sort_col is None or sort_col not in df.columns:
            raise ValueError(
                f"{task_name}: cannot find chronological sort column "
                f"(tried _sort_col + {TIMESERIES_FALLBACK_TIME_COL.get(task_name)})"
            )

    df = df.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    unique_ts = pd.Series(df[sort_col].unique()).sort_values().to_numpy()
    cut_idx = int(len(unique_ts) * (1 - STD_TEST_FRAC))
    cut_ts = unique_ts[cut_idx]
    train_mask = df[sort_col] < cut_ts
    train_frozen = df[train_mask].reset_index(drop=True)
    std_test = df[~train_mask].reset_index(drop=True)

    meta = {
        "task": task_name,
        "kind": "timeseries",
        "split_method": "chronological_tail",
        "sort_col": sort_col,
        "cut_value": cut_ts.item() if hasattr(cut_ts, "item") else str(cut_ts),
        "test_size": STD_TEST_FRAC,
        "train_frozen_rows": len(train_frozen),
        "std_test_rows": len(std_test),
        "target_col": data.target_col,
    }
    if dry_run:
        return meta

    train_frozen.to_parquet(os.path.join(out_dir, "train_frozen.parquet"))
    std_test.to_parquet(os.path.join(out_dir, "std_test.parquet"))
    _save_meta(out_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# Tabular graph (elliptic_bitcoin): hold out 20% of *labeled* nodes.
# ---------------------------------------------------------------------------
def build_graph(task_name, data, out_dir, dry_run):
    rng = np.random.default_rng(STD_TEST_SEED)
    feats = data.train_df
    classes = data.auxiliary_dfs.get("classes")
    if classes is None:
        raise ValueError(f"{task_name}: aux 'classes' df missing")

    labeled = classes[classes["class"].isin(["1", "2", 1, 2])]
    n_holdout = int(len(labeled) * STD_TEST_FRAC)
    perm = rng.permutation(len(labeled))
    std_ids = set(labeled.iloc[perm[:n_holdout]]["txId"].astype(str))

    is_std = feats["txId"].astype(str).isin(std_ids)
    train_frozen = feats[~is_std].reset_index(drop=True)
    std_test_feats = feats[is_std].reset_index(drop=True)
    # Attach labels for std-test rows only (raw class string).
    cls_map = classes.set_index(classes["txId"].astype(str))["class"]
    std_test_feats["class"] = std_test_feats["txId"].astype(str).map(cls_map)

    meta = {
        "task": task_name,
        "kind": "graph",
        "split_method": "node_random_holdout",
        "seed": STD_TEST_SEED,
        "test_size": STD_TEST_FRAC,
        "train_frozen_rows": len(train_frozen),
        "std_test_rows": len(std_test_feats),
        "target_col": data.target_col,
    }
    if dry_run:
        return meta

    train_frozen.to_parquet(os.path.join(out_dir, "train_frozen.parquet"))
    std_test_feats.to_parquet(os.path.join(out_dir, "std_test.parquet"))
    _save_meta(out_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# Rec leave-one-out: per-user last interaction goes to std-test.
# ---------------------------------------------------------------------------
def build_rec_loo(task_name, data, out_dir, dry_run):
    rng = np.random.default_rng(STD_TEST_SEED)
    df = data.interaction_df.copy().reset_index(drop=True)
    if "user_id" not in df.columns or "item_id" not in df.columns:
        raise ValueError(
            f"{task_name}: interaction_df must have user_id/item_id"
        )
    df["__row_idx__"] = np.arange(len(df))
    feat_cfg = getattr(data, "model_cfg", {}).get("feature", {}) or {}
    target_col = feat_cfg.get("target_col")
    label_rule = feat_cfg.get("label_rule", {}) or {}
    positive_label = label_rule.get("positive_label", 1)

    # Sort within each user by timestamp then row order. Tenrec has no
    # timestamp column at this level → fall back to row order only.
    sort_keys = ["user_id"]
    if "timestamp" in df.columns:
        sort_keys.append("timestamp")
    sort_keys.append("__row_idx__")
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # Need ≥2 interactions per user to LOO (1 stays in train).
    counts = df.groupby("user_id").size()
    eligible_users = counts[counts >= 2].index
    df_eligible = df[df["user_id"].isin(eligible_users)]
    if target_col and target_col in df_eligible.columns:
        df_eligible = df_eligible[df_eligible[target_col] == positive_label]
    if df_eligible.empty:
        raise ValueError(
            f"{task_name}: no eligible positive interactions for rec LOO "
            f"(target_col={target_col!r}, positive_label={positive_label!r})"
        )

    last_row_idx = (
        df_eligible.groupby("user_id").tail(1)["__row_idx__"].to_numpy()
    )
    holdout_set = set(last_row_idx.tolist())
    is_std = df["__row_idx__"].isin(holdout_set)
    interaction_frozen = df[~is_std].drop(columns="__row_idx__").reset_index(drop=True)
    std_test = df[is_std].drop(columns="__row_idx__").reset_index(drop=True)

    # Fixed negatives per std-test positive: sample 100 items from the
    # global item pool that the user has not interacted with in the
    # frozen interactions. Use a single rng + per-user item-set to keep
    # determinism without blowing up memory.
    if "item_id" in df.columns:
        all_items = pd.unique(df["item_id"].dropna()).tolist()
    else:
        all_items = []
    user_seen = (
        interaction_frozen.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    neg_records = []
    items_arr = np.array(all_items)
    n_items = len(items_arr)
    if n_items == 0:
        raise ValueError(f"{task_name}: no items found for negative sampling")
    for _, row in std_test.iterrows():
        u = row["user_id"]
        seen = user_seen.get(u, set())
        seen = set(seen) | {row["item_id"]}
        # Sample with replacement-rejection up to a budget; on rare
        # collision overflow take what we have.
        budget = REC_NUM_NEGATIVES * 8
        idx = rng.integers(0, n_items, size=budget)
        cand = items_arr[idx]
        cand = [int(c) if isinstance(c, np.integer) else c for c in cand if c not in seen]
        if len(cand) < REC_NUM_NEGATIVES:
            # Top-up by exhaustive shuffle if many collisions.
            extra = [it for it in all_items if it not in seen]
            rng.shuffle(extra)
            cand = list(cand) + extra
        cand = list(dict.fromkeys(cand))[:REC_NUM_NEGATIVES]
        for neg in cand:
            neg_records.append({"user_id": u, "item_id": neg})

    negatives = pd.DataFrame(neg_records, columns=["user_id", "item_id"])

    meta = {
        "task": task_name,
        "kind": "rec_loo",
        "split_method": "leave_one_out_per_user",
        "seed": STD_TEST_SEED,
        "num_negatives_per_positive": REC_NUM_NEGATIVES,
        "interaction_frozen_rows": len(interaction_frozen),
        "std_test_rows": len(std_test),
        "negatives_rows": len(negatives),
        "users_with_holdout": int(std_test["user_id"].nunique()),
    }
    if dry_run:
        return meta

    interaction_frozen.to_parquet(
        os.path.join(out_dir, "interaction_frozen.parquet")
    )
    std_test.to_parquet(os.path.join(out_dir, "std_test.parquet"))
    negatives.to_parquet(os.path.join(out_dir, "std_test_negatives.parquet"))
    _save_meta(out_dir, meta)
    return meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_for_task(task_name, dry_run, data_dir=None):
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {sorted(TASK_REGISTRY)}"
        )
    dotted, kind = TASK_REGISTRY[task_name]
    data_cls = _import_class(dotted)
    data = data_cls(data_dir=_task_data_dir(task_name, data_dir))
    cfg = _load_model_config(task_name)
    if hasattr(data, "set_model_config"):
        data.set_model_config(cfg)

    print("=" * 60)
    print(f"[{task_name}] kind={kind}")
    print("=" * 60)
    data.load_data()

    out_dir = _std_test_dir(task_name, data_dir=data_dir)
    if kind == "binary":
        meta = build_binary(task_name, data, out_dir, dry_run)
    elif kind == "timeseries":
        meta = build_timeseries(task_name, data, out_dir, dry_run)
    elif kind == "graph":
        meta = build_graph(task_name, data, out_dir, dry_run)
    elif kind == "rec_loo":
        meta = build_rec_loo(task_name, data, out_dir, dry_run)
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    canonical_protocol = (
        getattr(data, "canonical_data_protocol", None)
        or canonical_data_protocol(task_name)
    )
    if canonical_protocol:
        meta["canonical_data_protocol"] = copy.deepcopy(canonical_protocol)

    label_rule = cfg.get("feature", {}).get("label_rule")
    if label_rule:
        meta["label_rule"] = label_rule
    if (canonical_protocol or label_rule) and not dry_run:
        _save_meta(out_dir, meta)

    print(f"[{task_name}] meta: {json.dumps(meta, default=str)}")
    if dry_run:
        print(f"[{task_name}] dry-run (no files written)")
    else:
        print(f"[{task_name}] wrote -> {out_dir}")
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_names",
        type=str,
        default=None,
        help="Comma-separated task names. Default: all tasks.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print split summary only; do not write files.",
    )
    p.add_argument(
        "--gpu_id", type=int, default=-1,
        help="GPU index to use (-1 = CPU). Sets CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Optional dataset root. When set, task files are stored under "
            "<data_dir>/<task>/data and std_test under "
            "<data_dir>/<task>/std_test."
        ),
    )
    return p.parse_args()


def _resolve_device(gpu_id):
    if gpu_id is None or gpu_id < 0:
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            print(f"[warn] gpu_id={gpu_id} requested but CUDA not available; falling back to CPU.")
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda:0"


def main():
    args = parse_args()
    _resolve_device(args.gpu_id)
    if args.data_names:
        names = [n.strip() for n in args.data_names.split(",") if n.strip()]
    else:
        names = sorted(TASK_REGISTRY)

    summary = {}
    for name in names:
        try:
            summary[name] = run_for_task(name, args.dry_run, data_dir=args.data_dir)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            summary[name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, meta in summary.items():
        print(f"  {name}: {meta}")


if __name__ == "__main__":
    main()
