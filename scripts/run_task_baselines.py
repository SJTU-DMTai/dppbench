import argparse
import json
import os
import subprocess
import sys
import time


TASKS = [
    ("amazon_beauty", "DIN", "gpu"),
    ("kuairec", "DIN", "gpu"),
    ("movielens", "DIN", "gpu"),
    ("yelp", "DIN", "gpu"),
    ("elliptic_bitcoin", "GraphSAGE", "gpu"),
    ("beijing_air_quality", "LightGBM", "cpu"),
    ("berka", "LightGBM", "cpu"),
    ("bike_sharing", "LightGBM", "cpu"),
    ("bondora", "LightGBM", "cpu"),
    ("citibike_jc_hourly", "LightGBM", "cpu"),
    ("default_credit", "LightGBM", "cpu"),
    ("fraud_detection", "LightGBM", "cpu"),
    ("home_credit", "LightGBM", "cpu"),
    ("nyc_taxi_hourly", "LightGBM", "cpu"),
    ("polish_bankruptcy", "LightGBM", "cpu"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--skip", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    skip = set(args.skip)
    summary = []
    summary_path = os.path.join(args.output_dir, "baseline", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    for data_name, model_name, device_kind in TASKS:
        if data_name in skip:
            print(f"[skip] {data_name}/{model_name}", flush=True)
            summary.append({
                "data_name": data_name,
                "model": model_name,
                "status": "skipped",
            })
            continue

        gpu_id = args.gpu_id if device_kind == "gpu" else -1
        cmd = [
            sys.executable,
            "scripts/train_task_model.py",
            "--data_name", data_name,
            "--model", model_name,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--gpu_id", str(gpu_id),
        ]
        print("=" * 80, flush=True)
        print(f"[run] {data_name}/{model_name} gpu_id={gpu_id}", flush=True)
        print("[cmd] " + " ".join(cmd), flush=True)
        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start
        status = "ok" if result.returncode == 0 else "failed"
        print(
            f"[done] {data_name}/{model_name} status={status} "
            f"returncode={result.returncode} elapsed_sec={elapsed:.1f}",
            flush=True,
        )
        summary.append({
            "data_name": data_name,
            "model": model_name,
            "gpu_id": gpu_id,
            "status": status,
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
        })
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    failed = [row for row in summary if row.get("status") == "failed"]
    print("=" * 80, flush=True)
    print(f"[summary] wrote {summary_path}", flush=True)
    print(f"[summary] failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
