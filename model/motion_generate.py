"""Generate causal sensor sequences labelled with an exact future orientation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from host.orientation_sim import MODEL_DT_S
from model.broad_train import positive_integer, sha256
from model.orientation_generate import FAMILIES, nonnegative_integer, trajectory

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('model/motion_generate.py', 'model/orientation_generate.py', 'host/orientation_sim.py')


def task_identity(horizon_s):
    return {'kind': 'motion_prediction', 'horizon_s': horizon_s,
            'model_dt_s': MODEL_DT_S, 'target': 'absolute_body_to_enu_wxyz'}


def future_targets(quaternions, horizon_s):
    """Shortest-arc SLERP at t+horizon; discard frames lacking future truth."""
    offset = horizon_s / MODEL_DT_S
    lower = math.floor(offset)
    fraction = offset - lower
    if math.isclose(fraction, 0, abs_tol=1e-12):
        fraction = 0.
    count = len(quaternions) - lower - int(fraction > 0)
    if count <= 0:
        raise ValueError('Trajectory must be longer than the prediction horizon')
    left = np.asarray(quaternions[lower:lower + count], dtype=np.float64).copy()
    if fraction == 0:
        return left
    right = np.asarray(quaternions[lower + 1:lower + count + 1], dtype=np.float64).copy()
    dots = np.sum(left * right, axis=1)
    right[dots < 0] *= -1
    dots = np.clip(np.abs(dots), 0., 1.)
    angles = np.arccos(dots)
    small = angles < 1e-8
    result = (1 - fraction) * left + fraction * right
    moving = ~small
    result[moving] = (np.sin((1 - fraction) * angles[moving])[:, None] * left[moving]
                      + np.sin(fraction * angles[moving])[:, None] * right[moving]) / np.sin(angles[moving])[:, None]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def generate(args):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    horizon = args.horizon_ms / 1000
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Prediction horizon must be finite and positive')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split_index, (split, count) in enumerate((('train', args.train_sequences),
                                                 ('validation', args.validation_sequences),
                                                 ('confirmation', args.confirmation_sequences))):
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_index, index]))
            family = args.families[index % len(args.families)]
            sensors, poses = trajectory(rng, family, args.length)
            targets = future_targets(poses, horizon)
            sequence_id = f'{split}-{index:04d}'
            path = output / f'{sequence_id}.npz'
            np.savez_compressed(path, sensors=sensors[:len(targets)],
                                current_quaternion=poses[:len(targets)], quaternion=targets)
            entries.append({'split': split, 'sequence_id': sequence_id, 'family': family,
                            'path': path.name, 'sha256': sha256(path)})
    manifest = {'dt_s': MODEL_DT_S, 'seed': args.seed, 'length': args.length,
                'task': task_identity(horizon), 'sensor_source': 'virtual_body',
                'quaternion_convention': 'wxyz_body_to_ENU', 'sequences': entries,
                'sources': {name: sha256(ROOT / name) for name in SOURCES}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def load_sequences(directory, split):
    manifest = json.loads((directory / 'manifest.json').read_text())
    task = manifest.get('task', {})
    horizon = task.get('horizon_s')
    if (type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0
            or task != task_identity(horizon)):
        raise ValueError('Dataset must declare a supported future motion prediction task')
    for name in SOURCES:
        if manifest.get('sources', {}).get(name) != sha256(ROOT / name):
            raise ValueError('Dataset source SHA256 mismatch')
    identities = [entry['sequence_id'] for entry in manifest['sequences']]
    if len(set(identities)) != len(identities):
        raise ValueError('Dataset sequence IDs must be globally unique')
    sequences = []
    for entry in manifest['sequences']:
        if entry['split'] != split:
            continue
        path = directory / entry['path']
        if Path(entry['path']).is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Sequence path must stay inside the dataset')
        if sha256(path) != entry['sha256']:
            raise ValueError('Dataset sequence SHA256 mismatch')
        with np.load(path, allow_pickle=False) as saved:
            sensors, targets = saved['sensors'].copy(), saved['quaternion'].copy()
            current = saved['current_quaternion'].copy()
        if (sensors.ndim != 2 or sensors.shape[1] != 9 or not len(sensors)
                or targets.shape != (len(sensors), 4) or current.shape != targets.shape
                or not all(np.isfinite(values).all() for values in (sensors, targets, current))
                or not all(np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-8, rtol=0)
                           for values in (targets, current))):
            raise ValueError('Expected finite nine-channel sensors and unit quaternion targets/current pose')
        sequences.append((entry, sensors, targets))
    if not sequences:
        raise ValueError(f'Dataset has no {split} sequences')
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-sequences', type=nonnegative_integer, default=64)
    parser.add_argument('--validation-sequences', type=nonnegative_integer, default=16)
    parser.add_argument('--confirmation-sequences', type=nonnegative_integer, default=0)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--length', type=positive_integer, default=1000)
    parser.add_argument('--horizon-ms', type=float, default=50.)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    if args.train_sequences + args.validation_sequences + args.confirmation_sequences == 0:
        parser.error('At least one sequence is required')
    try:
        generate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Motion generation failed: {error}\n')


if __name__ == '__main__':
    main()
