"""Remote GPU comparison; each variant runs in a fresh interpreter.

All runs stop after --steps updates, never save a formal pipeline or run HPO.
Use --profile_steps 0 for wall-time comparisons; a separate traced pass measures
operators. Variants compare synchronization/data-placement ablations of current
v2, NOT the complete pre-change implementation. Data loading remains full-size.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--task', choices=['kuairec', 'yelp'], required=True)
    p.add_argument('--gpu_id', default='0')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--epochs', type=int, default=200, help='Upper epoch guard; --steps is the primary stop')
    p.add_argument('--profile_steps', type=int, default=0)
    p.add_argument('--variant', choices=['all', 'legacy_cpu', 'optimized_cpu', 'optimized_auto'], default='all')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.steps <= 0 or args.epochs <= 0 or not 0 <= args.profile_steps <= args.steps:
        p.error('steps/epochs must be positive; profile_steps must be between 0 and steps')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier diagnostic or experiment.
    if not args.worker:
        variants = ['legacy_cpu', 'optimized_cpu', 'optimized_auto'] if args.variant == 'all' else [args.variant]
        for variant in variants:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_id)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--data_dir', args.data_dir,
                       '--task', args.task, '--output_dir', str(out/variant), '--variant', variant,
                       '--epochs', str(args.epochs), '--steps', str(args.steps), '--profile_steps', str(args.profile_steps)]
            subprocess.run(command, env=env, cwd=ROOT, check=True)
        return
    sys.path.insert(0, str(ROOT))
    import torch
    from baselines.DAGPipe.dagpipe import DAGPipe
    if not torch.cuda.is_available():
        raise RuntimeError('This comparison requires the remote CUDA GPU; no silent CPU fallback')
    cfg = yaml.safe_load((ROOT/'baselines/DAGPipe/config.yaml').read_text())
    cfg.update(epochs=args.epochs, profile_steps=args.profile_steps, diagnostic_steps=args.steps,
               data_placement='auto' if args.variant=='optimized_auto' else 'cpu',
               sync_mode='legacy' if args.variant=='legacy_cpu' else 'optimized')
    config = out/'config.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    info = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                device=torch.cuda.get_device_name(0), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                note='Synchronization/placement ablation only. CPU means host-resident DATA; all compute on GPU. '
                     'Shared v2 architecture omits redundant DAG penalty; this is not a historical baseline.')
    (out/'hardware.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    runner = DAGPipe(task_dir=str(ROOT/'dppbench/tasks'/args.task), data_name=args.task,
                     data_dir=args.data_dir, device='cuda:0', model_name='DIN', max_pipeline_len=10,
                     config_path=str(config), output_dir=out, seed=42)
    result = runner.run()
    assert result['diagnostic_only'] and result['architecture_steps'] <= args.steps
    (out/'diagnostic_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
