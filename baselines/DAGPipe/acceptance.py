"""Small server acceptance: search/freeze/reload only; no HPO or test evaluation."""
import argparse
import json
from pathlib import Path
import torch
from baselines.common.budget import Budget, budget_scope
from baselines.common.search_cost import SearchCost, search_cost_scope
from .dagpipe import DAGPipe
from .pipeline import DAGPipePipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, choices=[1, 2], default=1)
    parser.add_argument('--max-memory-mb', type=float, default=71680)
    parser.add_argument('--max-gpu-memory-mb', type=float, default=81920)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = DAGPipe(str(root/'dppbench/tasks'/args.task), args.task, data_dir=args.data_dir,
                     seed=args.seed, output_dir=args.output_dir, device=args.device, max_pipeline_len=10)
    runner.cfg['epochs'] = args.epochs
    budget = Budget(max_pipeline_length=10, max_evaluations=1,
                    max_memory_mb=args.max_memory_mb, max_gpu_memory_mb=args.max_gpu_memory_mb)
    cost = SearchCost('DAGPipe')
    with budget_scope(budget), search_cost_scope(cost, args.output_dir):
        runner.run()
    pipeline = DAGPipePipeline.from_yaml((args.output_dir/'best_pipeline.yaml').read_text(encoding='utf-8'))
    log = json.loads((args.output_dir/'DAGPipe_search.json').read_text(encoding='utf-8'))
    report = dict(acceptance_only=True, task=args.task, seed=args.seed, torch=torch.__version__,
                  pipeline_hash=pipeline.hash(), steps=len(pipeline), ops=pipeline.op_names(),
                  training_rows=pipeline.payload['training_rows'], reference_rows=len(pipeline.payload['reference']), freeze_verification=log['freeze_verification'],
                  numerical_nodes=log['numerical_nodes'], search_cost=cost.snapshot(),
                  note='No downstream fit, HPO, or test evaluation. Search cost includes sampled reference forwards.')
    (args.output_dir/'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
