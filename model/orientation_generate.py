"""Generate independent virtual-body trajectories using the demo sensor physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from host.orientation_sim import MODEL_DT_S, advance_pose, normalize_quaternion, sensors_for_step
from model.broad_train import positive_integer, sha256


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['hold', 'single_axis', 'mixed', 'stop_go', 'spin']
SOURCES = ('model/orientation_generate.py', 'host/orientation_sim.py')


def body_rotation(pose, velocity):
    speed = float(np.linalg.norm(velocity))
    if speed == 0:
        return pose.copy()
    half_angle = speed * MODEL_DT_S / 2
    delta = np.concatenate(([np.cos(half_angle)], velocity / speed * np.sin(half_angle)))
    w, x, y, z = pose
    a, b, c, d = delta
    target = [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
              w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a]
    return np.asarray(advance_pose(pose, target))


def trajectory(rng, family, length):
    pose = normalize_quaternion(rng.normal(size=4))
    axis = normalize_quaternion(np.concatenate(([0.], rng.normal(size=3))))[1:]
    rate = float(rng.uniform(1.5, 5.5))
    sensors, quaternions = [], []
    mixed_axis = axis.copy()
    for step in range(length):
        velocity = np.zeros(3)
        if step >= min(4, length):
            if family == 'single_axis' and length // 4 <= step < 3 * length // 4:
                velocity = axis * rate
            elif family == 'mixed':
                if step % 32 == 4:
                    mixed_axis = rng.normal(size=3)
                    mixed_axis /= np.linalg.norm(mixed_axis)
                velocity = mixed_axis * rate
            elif family == 'stop_go' and (step // max(4, length // 8)) % 2:
                velocity = axis * rate
            elif family == 'spin':
                velocity = axis * 6
        previous = pose.copy()
        pose = body_rotation(pose, velocity)
        sensors.append(sensors_for_step(previous, pose))
        quaternions.append(pose.copy())
    return np.asarray(sensors), np.asarray(quaternions)


def generate(args):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split_index, (split, count) in enumerate((('train', args.train_sequences),
                                                 ('validation', args.validation_sequences),
                                                 ('confirmation', args.confirmation_sequences))):
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, split_index, index]))
            family = args.families[index % len(args.families)]
            sensors, quaternions = trajectory(rng, family, args.length)
            sequence_id = f'{split}-{index:04d}'
            path = output / f'{sequence_id}.npz'
            np.savez_compressed(path, sensors=np.asarray(sensors), quaternion=np.asarray(quaternions))
            entries.append({'split': split, 'sequence_id': sequence_id, 'family': family,
                            'path': path.name, 'sha256': sha256(path)})
    manifest = {'dt_s': MODEL_DT_S, 'seed': args.seed, 'length': args.length,
                'sensor_source': 'virtual_body', 'sequences': entries,
                'config': {'families': args.families, 'train_sequences': args.train_sequences,
                           'validation_sequences': args.validation_sequences,
                           'confirmation_sequences': args.confirmation_sequences},
                'max_angular_rate_rad_s': 6,
                'sensor_order': [f'{sensor}_{axis}' for sensor in ('acc', 'gyr', 'mag') for axis in 'xyz'],
                'sensor_units': {'acc': 'm/s^2', 'gyr': 'rad/s', 'mag': 'uT'},
                'quaternion_convention': 'wxyz_body_to_ENU',
                'sources': {name: sha256(ROOT / name) for name in SOURCES}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def load_sequences(directory, split):
    manifest = json.loads((directory / 'manifest.json').read_text())
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
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Sequence path must stay inside the dataset')
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Sequence path must stay inside the dataset')
        if sha256(path) != entry['sha256']:
            raise ValueError('Dataset sequence SHA256 mismatch')
        with np.load(path, allow_pickle=False) as saved:
            sensors, quaternion = saved['sensors'].copy(), saved['quaternion'].copy()
        if (sensors.ndim != 2 or sensors.shape[1] != 9 or not len(sensors)
                or quaternion.shape != (len(sensors), 4)
                or not np.isfinite(sensors).all() or not np.isfinite(quaternion).all()):
            raise ValueError('Expected finite nine-channel sensors and quaternion targets')
        if not np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-8, rtol=0):
            raise ValueError('Dataset quaternion targets must be unit length')
        sequences.append((entry, sensors, quaternion))
    if not sequences:
        raise ValueError(f'Dataset has no {split} sequences')
    return sequences


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('Count must be nonnegative')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-sequences', type=nonnegative_integer, default=64)
    parser.add_argument('--validation-sequences', type=nonnegative_integer, default=16)
    parser.add_argument('--confirmation-sequences', type=nonnegative_integer, default=0)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    parser.add_argument('--length', type=positive_integer, default=1000)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    if args.train_sequences + args.validation_sequences + args.confirmation_sequences == 0:
        parser.error('At least one sequence is required')
    try:
        generate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Orientation generation failed: {error}\n')


if __name__ == '__main__':
    main()
