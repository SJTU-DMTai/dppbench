"""Print imported source paths/hashes and probe evaluation without task data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.DAGPipe.preflight import check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_names', default='amazon_beauty,kuairec')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    for task in args.data_names.split(','):
        task = task.strip()
        print(json.dumps(check_runtime(task_dir=str(ROOT / 'dppbench/tasks' / task),
                                       data_name=task, model_name=args.model), indent=2))


if __name__ == '__main__':
    main()
