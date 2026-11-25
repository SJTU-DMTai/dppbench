import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.exps import run_llm_all_tasks as runner


class RunnerTest(unittest.TestCase):
    def dry_run(self, *options):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['runner', *options, '--dry_run']), contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(), 0)
        return output.getvalue().splitlines()

    def test_shared_repair_defaults_on_and_can_be_disabled(self):
        import shlex
        for options, flag in [((), '--shared_repair'), (('--shared_repair',), '--shared_repair'),
                              (('--no-shared_repair',), '--no-shared_repair')]:
            for line in self.dry_run(*options):
                command = shlex.split(line)
                method = command[command.index('--baseline') + 1]
                self.assertEqual(flag in command, method in ('BAT', 'DeepPrep', 'LLM-AutoDP-Adapted'))
                opposite = '--no-shared_repair' if flag == '--shared_repair' else '--shared_repair'
                self.assertNotIn(opposite, command)

    def test_policy_scoring_alignment(self):
        import shlex
        for line in self.dry_run():
            command = shlex.split(line)
            method = command[command.index('--baseline') + 1]
            limit = command[command.index('--max_evaluations') + 1]
            if method in runner.METHODS[:3]:
                self.assertEqual(limit, '100')
                self.assertEqual(command[command.index('--small_n') + 1], '3000')
                self.assertIn('--fast_train', command)
            else:
                self.assertEqual(limit, '1000')
                self.assertNotIn('--small_n', command)
        line = self.dry_run('--methods', 'CtxPipe', '--gpus', '0', '--policy_max_evaluations', '20',
                            '--small_n', '500', '--no-fast_train')[0]
        self.assertIn('--max_evaluations 20', line)
        self.assertIn('--small_n 500', line)
        self.assertIn('--no-fast_train', line)

    def test_defaults_and_task_ram_override(self):
        commands = self.dry_run()
        self.assertEqual(len(commands), 252)
        yelp = [line for line in commands if '--data_names yelp ' in line]
        self.assertEqual(len(yelp), 18)
        for line in yelp:
            expected = 229376 if '--baseline HAIPipe ' in line else 163840
            self.assertIn(f'--max_memory_mb {expected}', line)
            self.assertIn('--max_gpu_memory_mb 71680', line)
        selected_hai = self.dry_run('--methods', 'HAIPipe', '--gpus', '0', '--tasks', 'yelp',
                                    '--method_task_memory_mb', 'HAIPipe:yelp=245760')
        self.assertTrue(all('--max_memory_mb 245760' in line for line in selected_hai))
        selected = self.dry_run('--methods', 'DeepPrep', '--gpus', '6', '--tasks', 'yelp',
                                '--task_memory_mb', 'yelp=131072')
        self.assertEqual(len(selected), 3)
        self.assertTrue(all('--max_memory_mb 131072' in line for line in selected))

    def test_retry_later_run_overrides_completed_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            sources = [pathlib.Path(temp) / name for name in ('old', 'new')]
            for source in sources:
                source.mkdir()
            def entry(task, success):
                return dict(event='task_end', baseline='DeepPrep', seed=42, task=task, success=success)
            (sources[0] / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in
                [entry('amazon_beauty', False), entry('kuairec', True), entry('yelp', False)]))
            (sources[1] / 'events.jsonl').write_text(json.dumps(entry('amazon_beauty', True)))
            lines = self.dry_run('--methods', 'DeepPrep', '--gpus', '6', '--retry_from',
                                 *(str(p) for p in sources))
            self.assertEqual(len(lines), 1)
            self.assertIn('--data_names yelp', lines[0])
            self.assertIn('--seed 42', lines[0])

    def test_lifecycle_serializes_high_memory_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text('{}')
            output = root / 'run'
            counts = {'active': 0, 'peak': 0}
            lock = threading.Lock()
            class Process:
                pid = 1234
                def __init__(self, command, **kwargs):
                    with lock:
                        counts['active'] += 1
                        counts['peak'] = max(counts['peak'], counts['active'])
                    path = pathlib.Path(command[command.index('--output_csv') + 1])
                    path.write_text('std_test_metric,std_test_n_rows\n0.7,10\n')
                def wait(self, timeout=None):
                    time.sleep(0.01)
                    with lock:
                        counts['active'] -= 1
                    return 0
                def poll(self):
                    return 0
            options = ['runner', '--methods', 'DeepPrep', 'BAT', '--gpus', '6,7',
                       '--tasks', 'yelp', '--seeds', '42', '--output_dir', str(output),
                       '--deepprep_model_path', str(model), '--qwen3_model_path', str(model)]
            with patch.object(sys, 'argv', options), patch.object(runner.subprocess, 'run') as preflight, \
                 patch.object(runner.subprocess, 'Popen', Process), \
                 patch.object(runner.subprocess, 'check_output', side_effect=FileNotFoundError('git')), \
                 patch.object(runner.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            self.assertIn('--preflight_only', preflight.call_args.args[0])
            self.assertIn('--skip_build', preflight.call_args.args[0])
            self.assertEqual(counts['peak'], 1)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(row['success'] for row in summary))
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['expected'], 2)
            self.assertEqual(sum(e['event'] == 'memory_slot_acquired' for e in events), 2)

    def test_preflight_failure_does_not_launch_or_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp) / 'run'
            with patch.object(sys, 'argv', ['runner', '--output_dir', str(output)]), \
                 patch.object(runner.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'preflight')), \
                 patch.object(runner.subprocess, 'Popen') as process:
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.main()
                process.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
