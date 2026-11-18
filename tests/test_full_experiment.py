"""CPU-only queue/protocol and Stiefel QR regressions; no model imports."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_experiment', ROOT / 'run_full_experiment.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class ProtocolTests(unittest.TestCase):
    def test_plan_a_is_four_new_trainings_and_twenty_eight_evaluations(self):
        tasks = suite.build_tasks([0])
        self.assertEqual(sum(t['kind'] == 'train' for t in tasks), 4)
        self.assertEqual(sum(t['kind'] == 'eval' for t in tasks), 28)
        self.assertEqual({t['seed'] for t in tasks}, {0})
        positions = {t['id']: i for i, t in enumerate(tasks)}
        for t in tasks:
            if t.get('dependency'):
                self.assertLess(positions[t['dependency']], positions[t['id']])

    def test_every_learned_evaluation_uses_its_own_preceding_training(self):
        tasks = suite.build_tasks()
        self.assertEqual(sum(t['kind'] == 'train' for t in tasks), 12)
        self.assertEqual(sum(t['kind'] == 'eval' for t in tasks), 60)
        seen = {}
        for t in tasks:
            if t.get('rotate'):
                train = seen[t['dependency']]
                self.assertEqual((t['model'], t['stack'], t['seed']),
                                 (train['model'], train['stack'], train['seed']))
            if t.get('method') == 'gptq':
                self.assertEqual(t['model'], 'tinyllama')
            seen[t['id']] = t

    def test_deterministic_baselines_are_not_falsely_counted_as_three_seeds(self):
        tasks = suite.build_tasks()
        for model in ('tinyllama', 'mistral'):
            for stack in ('official', 'jittor'):
                for method, bits in [('float', 16), ('rtn', 16), ('rtn', 4)]:
                    values = [t for t in tasks if t['kind'] == 'eval' and t['model'] == model
                              and t['stack'] == stack and t['method'] == method and t['bits'] == bits
                              and not t['rotate']]
                    self.assertEqual([t['seed'] for t in values], [0])

    def test_shared_initialization_is_orthogonal_and_seed_specific(self):
        a = suite.hadamard(16, np.random.default_rng(0))
        b = suite.hadamard(16, np.random.default_rng(1))
        np.testing.assert_allclose(a.T @ a, np.eye(16), atol=1e-6)
        self.assertFalse(np.array_equal(a, b))


class QRRegressionTests(unittest.TestCase):
    def test_qr_retraction_uses_column_signs_like_upstream(self):
        path = ROOT / 'jittor_spinquant/rotation/optimizer.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_qr_retraction')
        env = {'np': np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
        value = np.random.default_rng(21).normal(size=(4, 4))
        q, r = np.linalg.qr(value.T)
        expected = (q * np.sign(np.diag(r))[None, :]).T
        np.testing.assert_allclose(env['_qr_retraction'](value), expected, atol=1e-14)


class RunnerFailureTests(unittest.TestCase):
    def test_failed_training_skips_its_dependents_and_preserves_other_cases(self):
        tasks = suite.build_tasks()
        failed = 'tinyllama_official_train_s0'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('logs', 'results', 'checks', 'figures', 'rotations'):
                (root / name).mkdir()
            suite.write_json(root / 'protocol.json', dict(models={'tinyllama': {}, 'mistral': {}},
                tasks=tasks, seeds=[0, 1, 2], torch_python='fake', jittor_python='fake', gpu_sample_interval=10))
            suite.write_json(root / 'status.json', dict(state='prepared', completed=[], failed=[], skipped=[]))
            class Child:
                def __init__(self, command, **kwargs):
                    task_id = command[command.index('--task') + 1]
                    self.returncode = 9 if task_id == failed else 0
                    if self.returncode:
                        return
                    if '--preflight' in command:
                        suite.write_json(root / 'checks' / (task_id + '.json'), dict(initial_fixed_probe_loss=2.0))
                        return
                    task = next(t for t in tasks if t['id'] == task_id)
                    value = dict(task, status='complete', **({'ppl': 8.0} if task['kind'] == 'eval' else {}))
                    suite.write_json(root / 'results' / (task_id + '.json'), value)
                    if task['kind'] == 'train':
                        (root / 'rotations' / task_id).mkdir()
                        (root / 'rotations' / task_id / 'loss.csv').write_text('step,loss\n0,2.0\n')
                def poll(self):
                    return self.returncode
            with patch.object(suite.subprocess, 'Popen', Child), patch.object(suite.shutil, 'disk_usage',
                    return_value=SimpleNamespace(free=10 * 1024 ** 3)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(suite.run(root), 1)
            state = json.loads((root / 'status.json').read_text())
            self.assertEqual(state['state'], 'finished_with_failures')
            self.assertEqual(state['failed'], [failed])
            self.assertEqual(len(state['skipped']), 4)
            self.assertEqual(len(state['completed']), 67)
            summary = json.loads((root / 'summary_status.json').read_text())
            self.assertEqual((summary['completed_train'], summary['completed_eval']), (11, 56))


if __name__ == '__main__':
    unittest.main()
