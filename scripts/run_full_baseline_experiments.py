import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "LightGBM": "berka default_credit fraud_detection home_credit polish_bankruptcy beijing_air_quality bike_sharing citibike_jc_hourly nyc_taxi_hourly".split(),
    "DIN": "amazon_beauty kuairec movielens yelp".split(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run full-training-pool finetuning and frozen-test evaluation for all 13 tasks.")
    parser.add_argument("--baselines", nargs="+", choices=("FlowPipe", "CtxPipe", "HAIPipe"),
                        default=["FlowPipe", "CtxPipe", "HAIPipe"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5", help="Physical GPU IDs, one per baseline/model group")
    parser.add_argument("--skip_haipipe_din", action="store_true", help="Exclude the four HAIPipe/DIN experiments")
    parser.add_argument("--flow_pretrained_dir", type=Path,
                        default=ROOT / "ckpts/pretrained/seed42_v3_20261026_213243")
    parser.add_argument("--pretrained_dir", type=Path,
                        default=ROOT / "ckpts/pretrained/seed42_v3_retry_20261029_211708",
                        help="CtxPipe and HAIPipe pretrained root")
    parser.add_argument("--qwen3_model_path", type=Path, default=ROOT / "ckpts/Qwen3-8B")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "dppbench/tasks")
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "outputs" / datetime.now().strftime("full_finetune_%Y%m%d_%H%M%S"))
    parser.add_argument("--training_mode", choices=("finetune", "pretrained"), default="finetune")
    parser.add_argument("--finetune_frames", type=int, default=2000)
    parser.add_argument("--finetune_epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def validate_result(folder):
    with (folder / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one result row: {folder}")
    row = rows[0]
    for key in ("error", "std_test_error", "baseline_eval_error"):
        if row.get(key):
            raise ValueError(f"{key}: {row[key]}")
    if int(row["std_test_n_rows"]) <= 0 or not math.isfinite(float(row["std_test_metric"])):
        raise ValueError(f"Invalid frozen-test result: {folder}")
    if not (folder / "best_pipeline.yaml").is_file():
        raise FileNotFoundError(folder / "best_pipeline.yaml")
    return row


def main():
    args = parse_args()
    gpus = args.gpus.split(",")
    jobs = [(baseline, model, tasks) for baseline in args.baselines for model, tasks in GROUPS.items()]
    if args.skip_haipipe_din:
        jobs = [job for job in jobs if job[:2] != ("HAIPipe", "DIN")]
    if len(set(args.baselines)) != len(args.baselines):
        raise ValueError("Duplicate baselines")
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError(f"Provide {len(jobs)} distinct physical GPU IDs")
    if min(args.finetune_frames, args.finetune_epochs, args.threads) <= 0:
        raise ValueError("Training budgets and threads must be positive")
    for baseline, model, tasks in jobs:
        pretrained = args.flow_pretrained_dir if baseline == "FlowPipe" else args.pretrained_dir
        source_model = "LightGBM" if baseline == "HAIPipe" and model == "DIN" else model
        folder = pretrained / baseline / source_model
        required = ["pretrain.json", "aipipe.pt" if baseline == "HAIPipe" else "checkpoint.pt"]
        if baseline == "HAIPipe":
            required.append("alternative_evaluator.pt")
        for filename in required:
            path = folder / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Pretraining is incomplete: {path}")
        with (folder / "pretrain.json").open() as stream:
            metadata = json.load(stream)["checkpoint_metadata"]
        if metadata.get("search_semantics") != "primary_schema_random_subset_rec_optional_fe_v2":
            raise ValueError(f"Incompatible search semantics: {folder}")
        encoder = metadata.get("encoder")
        if encoder and Path(encoder["path"]).resolve() != args.qwen3_model_path.resolve():
            raise ValueError(f"Qwen3 path differs from pretraining: {folder}")
        for task in tasks:
            frozen = args.data_dir / task / "std_test"
            train = "interaction_frozen.parquet" if model == "DIN" else "train_frozen.parquet"
            for filename in (train, "std_test.parquet", "meta.json"):
                if not (frozen / filename).is_file():
                    raise FileNotFoundError(frozen / filename)
    if any(b != "HAIPipe" for b in args.baselines) and not (args.qwen3_model_path / "config.json").is_file():
        raise FileNotFoundError(args.qwen3_model_path / "config.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "experiment.json").write_text(json.dumps(config, indent=2))
    stop = threading.Event()
    active = set()
    lock = threading.Lock()

    def terminate(signum, frame):
        stop.set()
        with lock:
            for process in list(active):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def run_group(gpu, baseline, model, tasks):
        results = []
        pretrained = args.flow_pretrained_dir if baseline == "FlowPipe" else args.pretrained_dir
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(PYTHONUNBUFFERED="1", DPPBENCH_LLM_QUANTIZATION="none")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(args.threads)
        for task in tasks:
            if stop.is_set():
                break
            folder = args.output_dir / baseline / f"{task}_{model}"
            folder.mkdir(parents=True)
            command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_baseline.py"),
                       "--baseline", baseline, "--data_names", task, "--model", model,
                       "--training_mode", args.training_mode, "--pretrained_dir", str(pretrained),
                       "--data_dir", str(args.data_dir), "--gpu_id", gpu, "--seed", str(args.seed),
                       "--small_n", "0", "--no-fast_train", "--skip_build",
                       "--finetune_frames", str(args.finetune_frames),
                       "--finetune_epochs", str(args.finetune_epochs),
                       "--output_dir", str(args.output_dir), "--output_csv", str(folder / "metrics.csv")]
            if baseline == "CtxPipe":
                command += ["--ctxpipe_gte_model_path", str(args.qwen3_model_path),
                            "--context_cache_dir", str(args.output_dir / "context_cache" / baseline / task)]
            elif baseline == "FlowPipe":
                command += ["--qwen3_model_path", str(args.qwen3_model_path)]
            (folder / "command.json").write_text(json.dumps(command, indent=2))
            print(f"START GPU={gpu} {baseline}/{task}/{model}", flush=True)
            try:
                with (folder / "launch.log").open("w") as log:
                    with lock:
                        if stop.is_set():
                            break
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        active.add(process)
                    try:
                        code = process.wait()
                    finally:
                        with lock:
                            active.discard(process)
                if code != 0:
                    raise RuntimeError(f"Evaluator exited with code {code}")
                row = validate_result(folder)
                status = {"baseline": baseline, "task": task, "model": model, "success": True, "result": row}
            except Exception as exc:
                status = {"baseline": baseline, "task": task, "model": model, "success": False,
                          "error": str(exc)}
            results.append(status)
            (folder / "status.json").write_text(json.dumps(status, indent=2))
            print(f"{'PASS' if status['success'] else 'FAIL'} {baseline}/{task}/{model}", flush=True)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_group, gpu, *job) for gpu, job in zip(gpus, jobs)]
        for future in as_completed(futures):
            results.extend(future.result())
            (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))
    successes = sum(result["success"] for result in results)
    expected = sum(len(tasks) for _, _, tasks in jobs)
    print(f"Completed: {len(results)}/{expected}; PASS={successes}; output={args.output_dir}", flush=True)
    return 0 if successes == expected else 1


if __name__ == "__main__":
    sys.exit(main())
