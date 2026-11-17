"""Linux cgroup launcher. Fail closed if systemd cannot enforce the limit.

python scripts/run_memory_limited.py --memory-gib 70 -- python scripts/evaluate_baseline.py ...
The scope covers the entire experiment AND descendants, including final HPO.
Requires a user systemd manager with delegated memory-controller permissions.
"""
import argparse
import os
import shutil
import subprocess
import sys


def build_command(memory_gib, command):
    if memory_gib <= 0:
        raise ValueError("memory-gib must be positive")
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Provide an experiment command after --")
    return ["systemd-run", "--user", "--scope",
            "-p", f"MemoryMax={int(memory_gib * 1024**3)}",
            "-p", "MemorySwapMax=0", "--", *command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gib", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not shutil.which("systemd-run"):
        parser.error("Linux systemd-run is required; refusing an unlimited fallback")
    try:
        command = build_command(args.memory_gib, args.command)
    except ValueError as exc:
        parser.error(str(exc))
    env = dict(os.environ, DPPBENCH_MEMORY_TRACE="1", PYTHONUNBUFFERED="1")
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
