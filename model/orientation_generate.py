"""Generate independent virtual-body trajectories using the demo sensor physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from host.orientation_sim import MODEL_DT_S, advance_pose, normalize_quaternion, sensors_for_step
from model.broad_train import positive_integer, sha256


def generate(args):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split_index, (split, count) in enumerate((('train', args.train_sequences),
                                                 ('validation', args.validation_sequences))):
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_index, index]))
            pose = normalize_quaternion(rng.normal(size=4))
            target = pose.copy()
            sensors, quaternions = [], []
            for step in range(args.length):
                if step % 10 == 4:
                    target = normalize_quaternion(rng.normal(size=4))
                previous = pose.copy()
                if step % 10 >= 4:
                    pose = np.asarray(advance_pose(pose, target))
                sensors.append(sensors_for_step(previous, pose))
                quaternions.append(pose.copy())
            sequence_id = f'{split}-{index:04d}'
            path = output / f'{sequence_id}.npz'
            np.savez_compressed(path, sensors=np.asarray(sensors), quaternion=np.asarray(quaternions))
            entries.append({'split': split, 'sequence_id': sequence_id,
                            'path': path.name, 'sha256': sha256(path)})
    manifest = {'dt_s': MODEL_DT_S, 'seed': args.seed, 'length': args.length,
                'sensor_source': 'virtual_body', 'sequences': entries}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def load_sequences(directory, split):
    manifest = json.loads((directory / 'manifest.json').read_text())
    sequences = []
    for entry in manifest['sequences']:
        if entry['split'] != split:
            continue
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Sequence path must stay inside the dataset')
        path = directory / relative
        if sha256(path) != entry['sha256']:
            raise ValueError('Dataset sequence SHA256 mismatch')
        with np.load(path, allow_pickle=False) as saved:
            sensors, quaternion = saved['sensors'].copy(), saved['quaternion'].copy()
        if (sensors.ndim != 2 or sensors.shape[1] != 9 or not len(sensors)
                or quaternion.shape != (len(sensors), 4)
                or not np.isfinite(sensors).all() or not np.isfinite(quaternion).all()):
            raise ValueError('Expected finite nine-channel sensors and quaternion targets')
        sequences.append((entry, sensors, quaternion))
    if not sequences:
        raise ValueError(f'Dataset has no {split} sequences')
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-sequences', type=positive_integer, default=64)
    parser.add_argument('--validation-sequences', type=positive_integer, default=16)
    parser.add_argument('--length', type=positive_integer, default=1000)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    try:
        generate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Orientation generation failed: {error}\n')


if __name__ == '__main__':
    main()
