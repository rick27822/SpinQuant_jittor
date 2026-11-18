#!/usr/bin/env python3
"""Prepare real WikiText-2 parquet files and shared GPTQ tokens; never train.

No network access occurs unless --download-wikitext or --model-id is specified.
The GPTQ tokenizer/sampling policy matches the recorded TinyLlama experiment.
The training/evaluation shell scripts consume the parquet directory directly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args):
    import numpy as np
    from datasets import load_dataset
    from transformers import LlamaTokenizerFast

    model, data, output = args.model.resolve(), args.wikitext_dir.resolve(), args.output.resolve()
    # Check destination before performing downloads or tokenization.
    targets = ['calibration_ids.npy', 'test_ids.npy', 'shared_data.json', 'input_sha256.json']
    if any((output / name).exists() for name in targets):
        raise FileExistsError(f'Prepared files already exist in {output}; use a new output directory.')
    if args.model_id:
        from huggingface_hub import snapshot_download
        if not args.model_revision:
            raise ValueError('--model-id requires --model-revision; the recorded TinyLlama revision is unknown.')
        if model.exists() and any(model.iterdir()):
            raise FileExistsError('Download the model into an empty directory to avoid mixing revisions.')
        snapshot_download(repo_id=args.model_id, revision=args.model_revision, local_dir=str(model))
    if not (model / 'config.json').is_file():
        raise FileNotFoundError(f'Local HuggingFace model config not found: {model / "config.json"}')
    if args.download_wikitext:
        from huggingface_hub import hf_hub_download
        import shutil
        data.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'test', 'validation'):
            name = f'{split}-00000-of-00001.parquet'
            if (data / name).exists():
                raise FileExistsError(f'Refusing to overwrite {data / name}')
            cached = hf_hub_download('Salesforce/wikitext',
                filename='wikitext-2-raw-v1/' + name, repo_type='dataset',
                revision=args.dataset_revision)
            shutil.copy2(cached, data / name)
    for split in ('train', 'test'):
        if not (data / f'{split}-00000-of-00001.parquet').is_file():
            raise FileNotFoundError(f'Missing {split} parquet in {data}; use --download-wikitext if needed.')
    if args.verify_recorded_tinyllama and (args.seed, args.seqlen, args.nsamples) != (0, 2048, 128):
        raise ValueError('Recorded TinyLlama verification requires seed=0, seqlen=2048, nsamples=128.')
    tokenizer = LlamaTokenizerFast.from_pretrained(str(model), local_files_only=True,
        model_max_length=args.seqlen, padding_side='right', add_eos_token=False, add_bos_token=False)
    dataset = load_dataset('parquet', data_files={
        split: str(data / f'{split}-00000-of-00001.parquet') for split in ('train', 'test')})
    train = tokenizer('\n\n'.join(dataset['train']['text']), return_tensors='np')['input_ids'].astype('int64')
    test = tokenizer('\n\n'.join(dataset['test']['text']), return_tensors='np')['input_ids'].astype('int64')
    if train.shape[1] < args.seqlen + 1 or test.size < args.seqlen:
        raise ValueError('WikiText-2 token streams are shorter than the requested sequence length.')
    rng = random.Random(args.seed)
    offsets = [rng.randint(0, train.shape[1] - args.seqlen - 1) for _ in range(args.nsamples)]
    calibration = np.stack([train[0, i:i + args.seqlen] for i in offsets])
    inputs = {'model/' + p.name: sha(p) for p in sorted(model.iterdir())
              if p.is_file() and p.suffix in ('.json', '.model', '.safetensors', '.bin')}
    inputs.update({'wikitext/' + p.name: sha(p) for p in sorted(data.glob('*.parquet'))})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='shared-') as temp:
        stage = Path(temp)
        np.save(stage / 'calibration_ids.npy', calibration)
        np.save(stage / 'test_ids.npy', test)
        metadata = dict(seed=args.seed, seqlen=args.seqlen, offsets=offsets,
            calibration_shape=list(calibration.shape), test_shape=list(test.shape),
            test_blocks=int(test.size // args.seqlen),
            evaluated_tokens=int(test.size // args.seqlen * (args.seqlen - 1)),
            tokenizer_class=type(tokenizer).__name__, tokenizer_add_bos=False, tokenizer_add_eos=False,
            calibration_sha256=sha(stage / 'calibration_ids.npy'), test_sha256=sha(stage / 'test_ids.npy'))
        if args.verify_recorded_tinyllama:
            reference = ROOT / 'experiments'
            shared = json.loads((reference / 'shared_sha256.json').read_text())
            sampling = json.loads((reference / 'shared/tinyllama_sampling_s0.json').read_text())
            # The ordinary entry saves a full stream; the controlled A runner
            # saves complete blocks. Compare identical blocked bytes here.
            blocks = test[0, :metadata['test_blocks'] * args.seqlen].reshape(-1, args.seqlen)
            np.save(stage / 'verification_test_blocks.npy', blocks)
            checks = {
                'calibration_sha256': metadata['calibration_sha256'] == shared['tinyllama_gptq_s0.npy'],
                'test_sha256': sha(stage / 'verification_test_blocks.npy') == shared['tinyllama_test.npy'],
                'offsets': offsets == sampling['gptq_offsets'],
                'test_blocks': metadata['test_blocks'] == 166,
                'evaluated_tokens': metadata['evaluated_tokens'] == 339802,
            }
            changed = [key for key, passed in checks.items() if not passed]
            if changed:
                raise ValueError('Recorded TinyLlama data mismatch: ' + ', '.join(changed))
            expected = json.loads((reference / 'input_sha256.json').read_text())
            required = ['model/config.json', 'model/model.safetensors', 'model/tokenizer.json',
                        'model/tokenizer_config.json', 'wikitext/train-00000-of-00001.parquet',
                        'wikitext/test-00000-of-00001.parquet']
            if any(inputs.get(k) != expected[k.replace('model/', 'tinyllama/', 1)] for k in required):
                raise ValueError('Model/tokenizer/data SHA256 differs from recorded TinyLlama inputs.')
        (stage / 'shared_data.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (stage / 'input_sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')
        output.mkdir(parents=True, exist_ok=True)
        import shutil
        for name in targets:
            shutil.copy2(stage / name, output / name)
    print(f'Prepared real data: calibration={calibration.shape}; test blocks={metadata["test_blocks"]}.')
    print(f'Shared arrays: {output}')
    print(f'Training/evaluation parquet directory (WIKITEXT2_DIR): {data}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Local HF model directory')
    parser.add_argument('--model-id', help='Optional HF repo to download; no default model substitution')
    parser.add_argument('--model-revision', help='Explicit revision for model download')
    parser.add_argument('--wikitext-dir', type=Path, required=True)
    parser.add_argument('--download-wikitext', action='store_true')
    parser.add_argument('--dataset-revision', default='main', help='Override to pin a dataset revision')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/shared_data')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verify-recorded-tinyllama', action='store_true')
    args = parser.parse_args(argv)
    if args.seqlen < 2 or args.nsamples < 1:
        parser.error('--seqlen must be >=2 and --nsamples must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
