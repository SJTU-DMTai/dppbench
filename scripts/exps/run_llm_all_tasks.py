#!/usr/bin/env python3
"""One GPU worker per method; aligned policy search followed by full-data final HPO."""
import argparse
import csv
import json
import math
import os
import signal
import shlex
import time
import traceback
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("HAIPipe", "CtxPipe", "FlowPipe", "BAT", "DeepPrep", "LLM-AutoDP-Adapted")
TASKS = "amazon_beauty,kuairec,movielens,yelp,elliptic_bitcoin,beijing_air_quality,berka,bike_sharing,citibike_jc_hourly,default_credit,fraud_detection,home_credit,nyc_taxi_hourly,polish_bankruptcy".split(",")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                   help="Methods to run; GPU IDs correspond to this order")
    p.add_argument("--gpus", default="0,1,2,3,4,5")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--retry_from", nargs="+", type=Path, default=[],
                   help="Retry failed task_end entries; later runs override earlier completed entries")
    p.add_argument("--data_dir", type=Path, default=ROOT / "dppbench/tasks")
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / datetime.now().strftime("llm_all_tasks_%Y%m%d_%H%M%S"))
    p.add_argument("--qwen3_model_path", type=Path, default=ROOT / "ckpts/Qwen3-8B")
    p.add_argument("--deepprep_model_path", type=Path, default=ROOT / "ckpts/DeepPrep-Qwen3-8B")
    p.add_argument("--pretrained_dir", type=Path, default=ROOT / "ckpts/pretrained/seed42_v3_retry_20261029_211708")
    p.add_argument("--flow_pretrained_dir", type=Path, default=ROOT / "ckpts/pretrained/seed42_v3_20261026_213243")
    p.add_argument("--training_mode", choices=("finetune", "pretrained", "scratch"), default="finetune")
    p.add_argument("--finetune_frames", type=int, default=None)
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--max_pipeline_len", type=int, default=10)
    p.add_argument("--max_evaluations", type=int, default=1000)
    p.add_argument("--policy_max_evaluations", type=int, default=100,
                   help="Shared search scoring cap for HAIPipe/CtxPipe/FlowPipe")
    p.add_argument("--small_n", type=int, default=3000)
    p.add_argument("--fast_train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--shared_repair', action=argparse.BooleanOptionalAction, default=True,
                   help='Shared completion for BAT/DeepPrep/LLMAutoDP (enabled by default)')
    p.add_argument("--max_llm_calls", type=int, default=100)
    p.add_argument("--max_memory_mb", type=int, default=71680)
    p.add_argument("--task_memory_mb", nargs="*", default=["yelp=163840"], metavar="TASK=MB",
                   help="Per-task host RAM limit; Yelp defaults to 160 GiB, GPU limit is unchanged")
    p.add_argument('--method_task_memory_mb', nargs='*', default=['HAIPipe:yelp=229376'],
                   metavar='METHOD:TASK=MB', help='Method-specific host RAM overrides; HAIPipe Yelp defaults to 224 GiB')
    p.add_argument("--max_gpu_memory_mb", type=int, default=71680)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--heartbeat_seconds", type=int, default=60, help="Process and output-log heartbeat interval")
    p.add_argument("--dry_run", action="store_true", help="Print all commands without writing files or launching experiments")
    return p.parse_args()


def memory_limit(args, method, task):
    return args.method_task_memory_mb.get(f'{method}:{task}', args.task_memory_mb.get(task, args.max_memory_mb))


def failure_kind(error):
    text = error or ''
    if 'unsupported' in text.lower() or 'does not support' in text.lower():
        return 'unsupported_task'
    if 'Pretrained agent missing' in text or 'checkpoint' in text.lower() and 'FileNotFoundError' in text:
        return 'missing_checkpoint'
    if 'memory budget exceeded' in text:
        return 'memory_budget'
    return 'execution_error' if text else None


def checkpoint_requirements(args, plan):
    if args.training_mode == 'scratch':
        return []
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from baselines.common.training_utils import resolve_policy_source
    issues = []
    for method, task in dict.fromkeys((method, task) for method, _, task in plan if method in METHODS[:3]):
        model = ('GraphSAGE' if task == 'elliptic_bitcoin' else
                 'DIN' if task in ('amazon_beauty', 'kuairec', 'movielens', 'yelp') else 'LightGBM')
        if method == 'HAIPipe' and model == 'DIN':
            model = 'LightGBM'
        root = args.flow_pretrained_dir if method == 'FlowPipe' else args.pretrained_dir
        try:
            _, source, _ = resolve_policy_source(method, model, root=root, training_mode=args.training_mode)
            files = ('aipipe.pt', 'alternative_evaluator.pt') if method == 'HAIPipe' else ('checkpoint.pt',)
            for name in files:
                path = Path(source).parent / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f'Pretrained agent missing: {path}')
        except FileNotFoundError as exc:
            issues.append(dict(baseline=method, task=task, failure_kind='missing_checkpoint', error=str(exc)))
    return issues


def graph_requirements(args, plan):
    return checkpoint_requirements(args, [item for item in plan if item[2] == 'elliptic_bitcoin'])


def command_for(args, method, task, seed, gpu):
    folder = args.output_dir / f"seed_{seed}" / method / task
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_baseline.py"),
               "--final_hpo", "--skip_build", "--baseline", method, "--data_names", task,
               "--data_dir", str(args.data_dir), "--output_dir", str(args.output_dir / f"seed_{seed}"),
               "--output_csv", str(folder / "results.csv"), "--seed", str(seed), "--gpu_id", gpu,
               "--ctxpipe_gte_model_path", str(args.qwen3_model_path),
               "--qwen3_model_path", str(args.qwen3_model_path),
               "--deepprep_model_path", str(args.deepprep_model_path)]
    for name in ("max_pipeline_len", "max_evaluations", "max_llm_calls", "max_memory_mb", "max_gpu_memory_mb"):
        value = memory_limit(args, method, task) if name == "max_memory_mb" else getattr(args, name)
        if name == "max_evaluations" and method in METHODS[:3]:
            value = args.policy_max_evaluations
        command += ["--" + name, str(value)]
    if method in METHODS[:3]:
        command += ["--small_n", str(args.small_n), "--fast_train" if args.fast_train else "--no-fast_train"]
        source = args.flow_pretrained_dir if method == "FlowPipe" else args.pretrained_dir
        command += ["--training_mode", args.training_mode, "--pretrained_dir", str(source)]
        for name in ("finetune_frames", "finetune_epochs"):
            if getattr(args, name) is not None:
                command += ["--" + name, str(getattr(args, name))]
    elif method == "LLM-AutoDP-Adapted":
        command += ["--llmautodp_model_path", str(args.qwen3_model_path)]
    if method in ('BAT', 'DeepPrep', 'LLM-AutoDP-Adapted'):
        command += ['--shared_repair' if args.shared_repair else '--no-shared_repair']
    return folder, command


def read_result(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one result row: {path}")
    row = rows[0]
    errors = [row[k] for k in ("error", "std_test_error", "baseline_eval_error") if row.get(k)]
    if errors:
        return row, "; ".join(errors)
    try:
        if int(row["std_test_n_rows"]) <= 0 or not math.isfinite(float(row["std_test_metric"])):
            return row, "Invalid frozen-test metric or row count"
    except (KeyError, ValueError, TypeError):
        return row, "Missing frozen-test metric or row count"
    return row, None


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    methods = args.methods
    if len(set(methods)) != len(methods):
        raise ValueError("Methods must be unique")
    gpus = args.gpus.split(",")
    if len(gpus) != len(methods) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError("Provide one distinct physical GPU ID per selected method")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if min(args.threads, args.heartbeat_seconds, args.max_pipeline_len, args.max_evaluations, args.max_llm_calls,
           args.max_memory_mb, args.max_gpu_memory_mb, args.policy_max_evaluations) <= 0:
        raise ValueError("Budgets and threads must be positive")
    if args.small_n < 0:
        raise ValueError("small_n must be nonnegative")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Tasks must be unique")
    memory_limits = {}
    for item in args.task_memory_mb:
        task, value = item.split("=", 1)
        if task not in TASKS or task in memory_limits or int(value) <= 0:
            raise ValueError(f"Invalid task memory override: {item}")
        memory_limits[task] = int(value)
    args.task_memory_mb = memory_limits
    method_limits = {}
    for item in args.method_task_memory_mb:
        key, value = item.split('=', 1)
        method, task = key.split(':', 1)
        if method not in METHODS or task not in TASKS or key in method_limits or int(value) <= 0:
            raise ValueError(f'Invalid method/task memory override: {item}')
        method_limits[key] = int(value)
    args.method_task_memory_mb = method_limits
    completed = {}
    for source in args.retry_from:
        for line in (source / "events.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "task_end":
                completed[(record["baseline"], record["seed"], record["task"])] = bool(record["success"])
    plan = [(method, seed, task) for method in methods for seed in args.seeds for task in args.tasks
            if not args.retry_from or completed.get((method, seed, task)) is False]
    if not plan:
        print("No failed tasks selected; nothing to run", flush=True)
        return 0
    if args.dry_run:
        assignment = dict(zip(methods, gpus))
        for method, seed, task in plan:
            print(shlex.join(command_for(args, method, task, seed, assignment[method])[1]))
        return 0
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/evaluate_baseline.py"),
                    "--preflight_only", "--skip_build", "--data_dir", str(args.data_dir),
                    "--data_names", ",".join(dict.fromkeys(task for _, _, task in plan))], check=True)
    model_paths = []
    if any(method in methods for method in ("CtxPipe", "FlowPipe", "BAT", "LLM-AutoDP-Adapted")):
        model_paths.append(args.qwen3_model_path)
    if "DeepPrep" in methods:
        model_paths.append(args.deepprep_model_path)
    for folder in model_paths:
        if not (folder / "config.json").is_file():
            raise FileNotFoundError(f"Complete local model required: {folder}")
    checkpoint_issues = checkpoint_requirements(args, plan)
    if checkpoint_issues:
        raise FileNotFoundError('Pretrained agent missing for planned tasks: ' + json.dumps(checkpoint_issues))
    graph_issues = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(retry_from=[str(path) for path in args.retry_from], methods=methods, tasks=args.tasks,
                  plan=plan, graph_preflight_issues=graph_issues, gpu_assignment=dict(zip(methods, gpus)),
                  search_sampling="aligned small_n/fast_train/scoring cap for policy methods; other methods native; final_hpo enabled",
                  evaluation_count_scope="cache-miss scoring attempts admitted by the search budget; failed attempts included; final HPO excluded",
                  source_pretrain_seed=42)
    config.update(code_commit=None, worktree_status=None)
    try:
        config["code_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.PIPE, timeout=15).strip()
        config["worktree_status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.PIPE, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        config["git_metadata_error"] = str(exc)
        print(f"WARNING: Git metadata unavailable; continuing experiments: {exc}", flush=True)
    (args.output_dir / "experiment.json").write_text(json.dumps(config, indent=2))
    active = set()
    lock = threading.Lock()
    stopped = threading.Event()
    results = []
    event_lock = threading.RLock()

    def event(kind, folder=None, **details):
        payload = {"timestamp": datetime.now().astimezone().isoformat(), "event": kind, **details}
        line = json.dumps(payload, ensure_ascii=False)
        with event_lock:
            with (args.output_dir / "events.jsonl").open("a") as stream:
                stream.write(line + "\n")
            if folder is not None:
                with (folder / "events.jsonl").open("a") as stream:
                    stream.write(line + "\n")
            print(line, flush=True)

    event("experiment_start", configuration=config, launcher_pid=os.getpid(), python=sys.executable)

    def save_summary():
        (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2))
        rows = [{"seed": r["seed"], "requested_baseline": r["baseline"], "requested_task": r["task"],
                 "success": r["success"], "launcher_error": r.get("error", ""), **r.get("metrics", {})}
                for r in results]
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def stop(signum, frame):
        stopped.set()
        event("stop_requested", signal=signum)
        with lock:
            for process in active:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    high_memory_slot = threading.Semaphore(1)

    def worker(method, gpu):
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(PYTHONUNBUFFERED="1", DPPBENCH_LLM_QUANTIZATION="none")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(args.threads)
        for seed in args.seeds:
            for task in args.tasks:
                if (method, seed, task) not in plan:
                    continue
                if stopped.is_set():
                    return
                folder, command = command_for(args, method, task, seed, gpu)
                folder.mkdir(parents=True)
                (folder / "command.json").write_text(json.dumps(command, indent=2))
                started = time.monotonic()
                status = {"baseline": method, "seed": seed, "task": task, "gpu": gpu, "success": False,
                          "started_at": datetime.now().astimezone().isoformat()}
                identity = {"baseline": method, "seed": seed, "task": task, "gpu": gpu}
                event("task_start", folder, **identity, command=shlex.join(command),
                      log=str(folder / "launch.log"), environment={k: env[k] for k in
                      ("PYTHONUNBUFFERED", "DPPBENCH_LLM_QUANTIZATION", "OMP_NUM_THREADS",
                       "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
                (folder / "status.json").write_text(json.dumps({**status, "state": "starting"}, indent=2))
                reserved = False
                try:
                    if memory_limit(args, method, task) > args.max_memory_mb:
                        event("memory_slot_wait", folder, **identity)
                        while not stopped.is_set():
                            reserved = high_memory_slot.acquire(timeout=1)
                            if reserved:
                                break
                        if not reserved:
                            return
                        event("memory_slot_acquired", folder, **identity)
                    with (folder / "launch.log").open("w") as log:
                        with lock:
                            if stopped.is_set():
                                return
                            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                            active.add(process)
                        status["pid"] = process.pid
                        event("process_started", folder, **identity, pid=process.pid)
                        (folder / "status.json").write_text(json.dumps({**status, "state": "running"}, indent=2))
                        try:
                            while True:
                                try:
                                    code = process.wait(timeout=args.heartbeat_seconds)
                                    break
                                except subprocess.TimeoutExpired:
                                    details = {"pid": process.pid, "elapsed_seconds": round(time.monotonic() - started, 1)}
                                    try:
                                        stat = (folder / "launch.log").stat()
                                        details.update(log_bytes=stat.st_size,
                                                       log_idle_seconds=round(time.time() - stat.st_mtime, 1))
                                        process_status = Path(f"/proc/{process.pid}/status").read_text()
                                        details["process_memory"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                                                                     for line in process_status.splitlines()
                                                                     if line.startswith(("VmRSS:", "VmHWM:", "Threads:", "State:"))}
                                    except OSError as exc:
                                        details["observation_error"] = str(exc)
                                    event("heartbeat", folder, **identity, **details)
                        finally:
                            with lock:
                                active.discard(process)
                    status["exit_code"] = code
                    if code:
                        raise RuntimeError(f"Evaluator exited {code}; see {folder / 'launch.log'}")
                    row, error = read_result(folder / "results.csv")
                    status["metrics"] = row
                    status["success"] = error is None
                    if error:
                        status["error"] = error
                except Exception as exc:
                    status["error"] = str(exc)
                    status["traceback"] = traceback.format_exc()
                    (folder / "launcher_error.log").write_text(status["traceback"])
                finally:
                    if reserved:
                        high_memory_slot.release()
                status.update(finished_at=datetime.now().astimezone().isoformat(),
                              elapsed_seconds=round(time.monotonic() - started, 3),
                              state="passed" if status["success"] else "failed",
                              failure_kind=None if status['success'] else failure_kind(status.get('error')))
                event("task_end", folder, **status)
                (folder / "status.json").write_text(json.dumps(status, indent=2))
                with lock:
                    results.append(status)
                    save_summary()
                print(f"{'PASS' if status['success'] else 'FAIL'} {method} seed={seed} task={task}", flush=True)

    with ThreadPoolExecutor(max_workers=len(methods)) as pool:
        futures = [pool.submit(worker, method, gpu) for method, gpu in zip(methods, gpus)]
        for future in as_completed(futures):
            future.result()
    expected = len(plan)
    passed = sum(r["success"] for r in results)
    event("experiment_end", recorded=len(results), expected=expected, passed=passed,
          failed=len(results) - passed, stopped=stopped.is_set(), output=str(args.output_dir))
    return 0 if passed == expected else 1


if __name__ == "__main__":
    sys.exit(main())
