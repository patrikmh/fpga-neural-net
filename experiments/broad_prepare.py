"""Convert declared BROAD splits; held-out conversion requires an explicit frozen spec."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(path, expected_rate):
    with h5py.File(path, 'r') as source:
        rate = float(source.attrs['sampling_rate'])
        channels = [np.asarray(source[name], dtype=np.float64)
                    for name in ('imu_acc', 'imu_gyr', 'imu_mag')]
        quaternion = np.asarray(source['opt_quat'], dtype=np.float64)
        movement = np.asarray(source['movement'])
    if not np.isfinite(rate) or not np.isclose(rate, expected_rate, rtol=1e-9, atol=0):
        raise ValueError(f'Sampling rate mismatch: {path.name}')
    count = len(channels[0])
    if count == 0 or any(values.shape != (count, 3) for values in channels):
        raise ValueError(f'Expected aligned three-axis sensor arrays: {path.name}')
    sensors = np.concatenate(channels, axis=1)
    if not np.isfinite(sensors).all():
        raise ValueError(f'Nonfinite sensor values: {path.name}')
    if quaternion.shape != (count, 4) or movement.shape != (count,):
        raise ValueError(f'Reference or movement dimensions mismatch: {path.name}')
    if not np.isfinite(movement).all() or not np.isin(movement, [False, True]).all():
        raise ValueError(f'Invalid movement mask: {path.name}')
    valid = np.isfinite(quaternion).all(axis=1) & np.any(quaternion != 0, axis=1)
    normalized = np.tile(np.array([1., 0., 0., 0.]), (count, 1))
    scaled = quaternion[valid] / np.max(np.abs(quaternion[valid]), axis=1, keepdims=True)
    normalized[valid] = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    return {'sensors': sensors, 'quaternion': normalized, 'movement': movement.astype(bool),
            'reference_valid': valid, 'sample_index': np.arange(count, dtype=np.int64),
            'rate_hz': np.array(rate)}


def prepare(data_dir, spec_path, output, split='train-validation'):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory must be empty: {output}')
    spec = json.loads(spec_path.read_text())
    if split == 'test':
        if not all(key in spec for key in ('checkpoint_sha256', 'training_manifest_sha256',
                                          'selection_sha256', 'sources', 'versions')):
            raise ValueError('Test conversion requires a frozen selection specification')
        if spec['sources'].get('experiments/broad_prepare.py') != sha256(Path(__file__)):
            raise ValueError('Frozen converter source SHA256 mismatch')
    selected = []
    ids, files = set(), set()
    for trial in spec['trials']:
        trial_id = int(trial['id'])
        filename = Path(trial['file'])
        if trial_id in ids or str(filename) in files:
            raise ValueError('Trial IDs and files must be unique across splits')
        ids.add(trial_id)
        files.add(str(filename))
        if trial['split'] not in ('train', 'validation', 'test'):
            raise ValueError(f"Unknown split: {trial['split']}")
        if filename.is_absolute() or '..' in filename.parts:
            raise ValueError('Trial filenames must remain inside the data directory')
        if not np.isfinite(trial['rate_hz']) or trial['rate_hz'] <= 0:
            raise ValueError('Trial sampling rates must be finite and positive')
        if (split == 'test' and trial['split'] == 'test') or (split != 'test' and trial['split'] != 'test'):
            selected.append((trial, trial_id))
    inputs = {}
    for trial, _ in selected:
        path = data_dir / trial['file']
        digest = sha256(path)
        if digest != trial['sha256']:
            raise ValueError(f"Input SHA256 mismatch: {trial['file']}")
        inputs[trial['file']] = digest
    converted = [(trial, trial_id, convert(data_dir / trial['file'], trial['rate_hz']))
                 for trial, trial_id in selected]
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for trial, trial_id, arrays in converted:
        filename = f"{trial['split']}/{trial_id:02d}.npz"
        path = output / filename
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        outputs[filename] = sha256(path)
    manifest = {
        'spec': spec, 'spec_sha256': sha256(spec_path), 'inputs': inputs, 'outputs': outputs,
        'sources': {'experiments/broad_prepare.py': sha256(Path(__file__))},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'h5py': h5py.__version__},
        'sensor_order': ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_gyr_x', 'imu_gyr_y',
                         'imu_gyr_z', 'imu_mag_x', 'imu_mag_y', 'imu_mag_z'],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'trials_converted': len(outputs)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train-validation', 'test'), default='train-validation')
    args = parser.parse_args()
    try:
        prepare(args.data_dir, args.spec, args.output_dir, args.split)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'BROAD preparation failed: {error}\n')


if __name__ == '__main__':
    main()
