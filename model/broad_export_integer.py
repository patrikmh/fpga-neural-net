"""Export a source-bound BROAD checkpoint as Q11/Q13 integer memories."""

import argparse
import json
import math
from pathlib import Path
import shutil

import torch

from model.broad_integer import NUMERIC, FILES, artifact_sources, runtime_versions, sha256, validate_checkpoint


def half_away(value):
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def export(checkpoint_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    size = validate_checkpoint(checkpoint)
    sources = artifact_sources(checkpoint)
    state, ordered = checkpoint['state_dict'], []
    for head in ('ff1', 'ff2', 'time_a', 'time_b'):
        if state[f'cell.{head}.weight'].shape != (size, 9 + size) or state[f'cell.{head}.bias'].shape != (size,):
            raise ValueError('Checkpoint affine shape mismatch')
        for row in range(size):
            ordered.extend(state[f'cell.{head}.weight'][row].tolist())
            ordered.append(float(state[f'cell.{head}.bias'][row]))
    if state['readout.weight'].shape != (4, size) or state['readout.bias'].shape != (4,):
        raise ValueError('Checkpoint readout shape mismatch')
    for row in range(4):
        ordered.extend(state['readout.weight'][row].tolist())
        ordered.append(float(state['readout.bias'][row]))
    weights = [half_away(value * 8192) for value in ordered]
    if any(value < -32768 or value > 32767 for value in weights):
        raise ValueError('Checkpoint coefficients overflow signed Q13')
    grid = [(index - 2048) / 256 for index in range(4096)]
    tables = {'weights.hex': weights,
              'tanh.hex': [half_away(math.tanh(value) * 8192) for value in grid],
              'sigmoid.hex': [half_away(8192 / (1 + math.exp(-value))) for value in grid]}
    checkpoint_hash = sha256(checkpoint_path)
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in tables.items():
        (output / filename).write_text(''.join(f'{value & 65535:04x}\n' for value in values))
    shutil.copyfile(checkpoint_path, output / 'float_checkpoint.pt')
    if sha256(checkpoint_path) != checkpoint_hash or sha256(output / 'float_checkpoint.pt') != checkpoint_hash:
        raise ValueError('Checkpoint identity changed during export')
    manifest = {'numeric': NUMERIC, 'checkpoint_sha256': checkpoint_hash,
                'config': checkpoint['config'], 'normalization': checkpoint['normalization'],
                'files': {name: sha256(output / name) for name in FILES},
                'sources': sources, 'versions': runtime_versions(),
                'weight_order': 'ff1,ff2,time_a,time_b: H rows of 9 inputs,H hidden,bias; readout: 4 rows of H hidden,bias',
                'vector_order': 'nine inputs, H new hidden states, four raw outputs; signed16 words, most significant first'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'coefficients': len(weights)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'BROAD integer export failed: {error}\n')


if __name__ == '__main__':
    main()
