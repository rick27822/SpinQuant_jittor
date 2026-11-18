"""CPU checks for the actual published evidence and preserved experiment math."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import importlib.util
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments'


class PublicationTests(unittest.TestCase):
    def test_data_preparation_verifies_recorded_blocked_reference(self):
        # Exercise the public verification flag with tiny fake files/tokenizer,
        # while retaining the actual experiment block count and GPTQ sample shape.
        # This checks the published path/schema and full-stream vs block format.
        spec = importlib.util.spec_from_file_location('prepare_release_data', ROOT / 'prepare_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, data = root / 'model', root / 'data'
            reference = root / 'experiments'
            for directory in (model, data, reference / 'shared'):
                directory.mkdir(parents=True)
            for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
                (model / name).write_text('{}')
            for name in ('train', 'test'):
                (data / f'{name}-00000-of-00001.parquet').write_text('fake parquet')
            inputs = {'tinyllama/' + p.name: module.sha(p) for p in model.iterdir()}
            inputs.update({'wikitext/' + p.name: module.sha(p) for p in data.iterdir()})
            (reference / 'input_sha256.json').write_text(json.dumps(inputs))
            train = np.arange(4097, dtype=np.int64)[None, :]
            test = np.arange(166 * 2048 + 17, dtype=np.int64)[None, :]
            rng = random.Random(0)
            offsets = [rng.randint(0, train.shape[1] - 2048 - 1) for _ in range(128)]
            np.save(root / 'cal.npy', np.stack([train[0, i:i+2048] for i in offsets]))
            np.save(root / 'test.npy', test[0, :166*2048].reshape(166, 2048))
            (reference / 'shared_sha256.json').write_text(json.dumps({
                'tinyllama_gptq_s0.npy': module.sha(root / 'cal.npy'),
                'tinyllama_test.npy': module.sha(root / 'test.npy')}))
            (reference / 'shared/tinyllama_sampling_s0.json').write_text(json.dumps({'gptq_offsets': offsets}))
            class Tokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return cls()
                def __call__(self, text, **kwargs):
                    return {'input_ids': train if text == 'train' else test}
            args = SimpleNamespace(model=model, wikitext_dir=data, output=root / 'out',
                model_id=None, download_wikitext=False, verify_recorded_tinyllama=True,
                seed=0, seqlen=2048, nsamples=128)
            dataset_module = SimpleNamespace(load_dataset=lambda *args, **kwargs:
                {'train': {'text': ['train']}, 'test': {'text': ['test']}})
            with contextlib.redirect_stdout(io.StringIO()), patch.object(module, 'ROOT', root), patch.dict(sys.modules, {
                'datasets': dataset_module, 'transformers': SimpleNamespace(LlamaTokenizerFast=Tokenizer)}):
                module.prepare(args)
                self.assertEqual(np.load(args.output / 'test_ids.npy').shape, test.shape)
                self.assertEqual(module.sha(args.output / 'calibration_ids.npy'), module.sha(root / 'cal.npy'))
                (model / 'model.safetensors').write_text('changed model')
                args.output = root / 'out_bad_model'
                with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                    module.prepare(args)

    def test_evidence_and_derived_csvs_are_consistent_without_gpu_or_numpy(self):
        result = subprocess.run([sys.executable, '-I', str(RUN / 'summarize.py'), '--check'],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual((report['train'], report['eval'], report['loss_records'], report['performance_records']),
                         (4, 28, 400, 32))

    def test_published_jittor_sources_match_the_measured_version(self):
        frozen = RUN / 'source/jittor/jittor_spinquant'
        for path in sorted((ROOT / 'jittor_spinquant').rglob('*')):
            if not path.is_file() or path.suffix not in ('.py', '.npz') or any(
                    x in path.parts for x in ('__pycache__', '.ipynb_checkpoints')):
                continue
            other = frozen / path.relative_to(ROOT / 'jittor_spinquant')
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(),
                                 hashlib.sha256(other.read_bytes()).digest())

    def test_portable_launcher_preserves_training_and_evaluation_math(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef)}
        current = functions(ROOT / 'run_full_experiment.py')
        measured = functions(RUN / 'source/run_full_experiment.py')
        for name in ('build_tasks', 'hadamard', 'quant_flags', 'rotation_file', 'load_backend',
                     'train_one', 'evaluate_one', 'worker', 'summarize'):
            with self.subTest(function=name):
                self.assertEqual(current[name], measured[name])


if __name__ == '__main__':
    unittest.main()
