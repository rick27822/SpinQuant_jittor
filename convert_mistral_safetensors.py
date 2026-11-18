#!/usr/bin/env python3
"""CPU-only, lossless format conversion of local Mistral .bin shards.

Use the PyTorch environment. Original files are read-only. An existing output
directory is refused. Tensor dtypes/values are preserved; the A-run preparer
independently verifies every tensor after the experiment's FP16 cast.
"""
import argparse
from pathlib import Path
import shutil


def convert(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Use a new output directory: {destination}')
    shards = sorted(source.glob('pytorch_model*.bin'))
    if not shards or not (source / 'config.json').is_file():
        raise FileNotFoundError(f'Missing Mistral config.json or pytorch_model*.bin shards in {source}')
    import torch
    from safetensors.torch import save_file
    state = {}
    for shard in shards:
        print('Loading', shard, flush=True)
        part = torch.load(shard, map_location='cpu', weights_only=True)
        overlap = state.keys() & part.keys()
        if overlap:
            raise ValueError('Duplicate tensor names across shards: ' + ', '.join(sorted(overlap)))
        state.update({k: v.contiguous() for k, v in part.items()})
    destination.mkdir(parents=True)
    for name in ('config.json', 'configuration.json', 'generation_config.json',
                 'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model',
                 'tokenizer_config.json', 'README.md', 'LICENSE', '.gitattributes'):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    save_file(state, str(destination / 'model.safetensors'), metadata={'format': 'pt'})
    print(f'Saved {len(state)} tensors to {destination / "model.safetensors"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
