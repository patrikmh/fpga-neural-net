"""Identity-bound held-out BROAD evaluation with causal sensor-only baselines."""

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.broad_network import BroadPredictor
from model.broad_train import predict_trial, sha256


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ('accelerometer_only', 'gyro_only_acc_initialization', 'gyro_acc_complementary')


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Scored vectors must be finite and nonzero')
    return values / norms


def orientation_metrics(predicted, target, mask, heading_epsilon):
    predicted, target = np.asarray(predicted), np.asarray(target)
    mask = np.asarray(mask, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 4 or mask.shape != predicted.shape[:1]:
        raise ValueError('Orientation dimensions mismatch')
    if not mask.any():
        raise ValueError('No scored orientation samples')
    if not np.isfinite(heading_epsilon) or heading_epsilon <= 0:
        raise ValueError('Heading epsilon must be finite and positive')
    p, q = unit_rows(predicted[mask]), unit_rows(target[mask])
    w = p[:, 0] * q[:, 0] + np.sum(p[:, 1:] * q[:, 1:], axis=1)
    v = -p[:, :1] * q[:, 1:] + q[:, :1] * p[:, 1:] - np.cross(p[:, 1:], q[:, 1:])
    heading_norm = np.sqrt(w * w + v[:, 2] * v[:, 2])
    full = np.degrees(2 * np.arccos(np.clip(np.abs(w), 0, 1)))
    inclination = np.degrees(2 * np.arccos(np.clip(heading_norm, 0, 1)))
    heading_valid = heading_norm > heading_epsilon
    heading = np.degrees(2 * np.arccos(np.clip(np.abs(w[heading_valid]) / heading_norm[heading_valid], 0, 1)))
    return {'full_rmse_deg': float(np.sqrt(np.mean(full ** 2))),
            'inclination_rmse_deg': float(np.sqrt(np.mean(inclination ** 2))),
            'heading_rmse_deg': float(np.sqrt(np.mean(heading ** 2))) if len(heading) else None,
            'samples': int(mask.sum()), 'heading_samples': int(heading_valid.sum()),
            'heading_singular_samples': int((~heading_valid).sum())}


def quaternion_up(quaternion):
    q = unit_rows(quaternion)
    w, x, y, z = q.T
    return np.column_stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def propagate(up, omega, dt):
    rate = -omega
    speed = float(np.linalg.norm(rate))
    if speed < 1e-12:
        return up.copy()
    axis, angle = rate / speed, speed * dt
    return up * np.cos(angle) + np.cross(axis, up) * np.sin(angle) + axis * np.dot(axis, up) * (1 - np.cos(angle))


def sensor_baselines(sensors, dt, tau):
    measured = unit_rows(sensors[:, :3])
    gyro, fused = np.empty_like(measured), np.empty_like(measured)
    gyro[0] = fused[0] = measured[0]
    alpha = 1 - np.exp(-dt / tau)
    for step in range(1, len(sensors)):
        omega = (sensors[step - 1, 3:6].astype(np.float64) + sensors[step, 3:6].astype(np.float64)) / 2
        gyro[step] = propagate(gyro[step - 1], omega, dt)
        gyro[step] /= np.linalg.norm(gyro[step])
        fused[step] = (1 - alpha) * propagate(fused[step - 1], omega, dt) + alpha * measured[step]
        fused[step] /= np.linalg.norm(fused[step])
    return measured, gyro, fused


def current_identities():
    sources = {name: sha256(ROOT / name) for name in
               ('model/broad_train.py', 'model/broad_network.py', 'model/broad_evaluate.py', 'experiments/broad_prepare.py')}
    sources['ncps/torch/cfc_cell.py'] = sha256(Path(inspect.getfile(CfCCell)))
    versions = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                'numpy': np.__version__, 'ncps': importlib.metadata.version('ncps')}
    return sources, versions


def evaluate(checkpoint_path, prepared, spec_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    spec = json.loads(spec_path.read_text())
    for key in ('checkpoint_sha256', 'training_manifest_sha256', 'selection_sha256'):
        value = spec[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid frozen SHA256 identity: {key}')
    if sha256(checkpoint_path) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 identity mismatch')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if checkpoint['prepared_manifest_sha256'] != spec['training_manifest_sha256']:
        raise ValueError('Training manifest SHA256 identity mismatch')
    sources, versions = current_identities()
    training_sources = {name: sources[name] for name in
                        ('model/broad_train.py', 'model/broad_network.py', 'ncps/torch/cfc_cell.py')}
    if checkpoint['provenance']['sources'] != training_sources or spec['sources'] != sources:
        raise ValueError('Frozen source SHA256 identity mismatch')
    if checkpoint['provenance']['versions'] != versions or spec['versions'] != versions:
        raise ValueError('Runtime versions identity mismatch')
    if type(spec['warmup']) is not int or spec['warmup'] < 0 or spec['warmup'] != checkpoint['config']['warmup']:
        raise ValueError('Frozen/checkpoint warmup mismatch')
    if spec['baseline_tau_s'] != 0.5 or not np.isfinite(spec['heading_epsilon']) or spec['heading_epsilon'] <= 0:
        raise ValueError('Invalid frozen baseline or heading configuration')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if manifest['spec'] != spec or manifest['spec_sha256'] != sha256(spec_path):
        raise ValueError('Prepared/frozen spec SHA256 binding mismatch')
    if manifest['sources']['experiments/broad_prepare.py'] != sources['experiments/broad_prepare.py']:
        raise ValueError('Prepared converter source SHA256 identity mismatch')
    trials = [trial for trial in spec['trials'] if trial['split'] == 'test']
    ids = [int(trial['id']) for trial in trials]
    files = [trial['file'] for trial in trials]
    if not trials or len(set(ids)) != len(ids) or len(set(files)) != len(files):
        raise ValueError('Test trial IDs/files must be nonempty and unique')
    if manifest['inputs'] != {trial['file']: trial['sha256'] for trial in trials}:
        raise ValueError('Prepared raw input SHA256 binding mismatch')
    expected_outputs = {f'test/{trial_id:02d}.npz' for trial_id in ids}
    if set(manifest['outputs']) != expected_outputs:
        raise ValueError('Prepared test trial IDs mismatch')
    for filename in expected_outputs:
        if sha256(prepared / filename) != manifest['outputs'][filename]:
            raise ValueError(f'Prepared test NPZ SHA256 mismatch: {filename}')
    mean, std = (np.asarray(checkpoint['normalization'][name]) for name in ('mean', 'std'))
    if mean.shape != (9,) or std.shape != (9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid frozen training normalization')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    network = BroadPredictor().eval()
    network.load_state_dict(checkpoint['state_dict'])
    results, saved_trials = [], []
    for entry in trials:
        trial_id = int(entry['id'])
        with np.load(prepared / f'test/{trial_id:02d}.npz', allow_pickle=False) as saved:
            trial = {name: saved[name].copy() for name in
                     ('sensors', 'quaternion', 'movement', 'reference_valid', 'sample_index')}
        sensors, target = trial['sensors'], trial['quaternion']
        count = len(sensors)
        if sensors.shape != (count, 9) or target.shape != (count, 4) or not np.isfinite(sensors).all():
            raise ValueError('Invalid test sensor/reference dimensions or values')
        if any(trial[name].shape != (count,) for name in ('movement', 'reference_valid', 'sample_index')):
            raise ValueError('Invalid test masks/index dimensions')
        rate = entry['rate_hz']
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Invalid frozen trial sampling rate')
        predicted = predict_trial(network, sensors, checkpoint['normalization'])
        mask = (trial['movement'] & trial['reference_valid']).copy()
        mask[:spec['warmup']] = False
        metrics = orientation_metrics(predicted, target, mask, spec['heading_epsilon'])
        reference_safe = target.copy()
        reference_safe[~trial['reference_valid']] = [1, 0, 0, 0]
        reference_up = quaternion_up(reference_safe)
        baseline_arrays = sensor_baselines(sensors, 1 / rate, spec['baseline_tau_s'])
        baselines = {}
        for name, up in zip(BASELINES, baseline_arrays):
            error = np.degrees(np.arccos(np.clip(np.sum(up[mask] * reference_up[mask], axis=1), -1, 1)))
            baselines[name] = {'incl_rmse_deg': float(np.sqrt(np.mean(error ** 2)))}
        filename = f'predictions/{trial_id:02d}.npz'
        results.append({'trial_id': trial_id, 'predictions_file': filename, 'cfc': metrics, 'baselines': baselines})
        saved_trials.append((filename, dict(sensors=sensors, targets=target, predictions=predicted,
            movement=trial['movement'], reference_valid=trial['reference_valid'], sample_index=trial['sample_index'],
            predicted_up=quaternion_up(predicted), reference_up=reference_up,
            accelerometer_up=baseline_arrays[0], gyro_up=baseline_arrays[1], complementary_up=baseline_arrays[2])))
        print(f"trial {trial_id}: full RMSE {metrics['full_rmse_deg']:.5f} deg", file=sys.stderr, flush=True)
    mean_cfc = {}
    for name in ('full_rmse_deg', 'inclination_rmse_deg', 'heading_rmse_deg'):
        values = [result['cfc'][name] for result in results if result['cfc'][name] is not None]
        mean_cfc[name] = float(np.mean(values)) if values else None
    means = {'cfc': mean_cfc, 'baselines': {name: {'incl_rmse_deg': float(np.mean(
        [result['baselines'][name]['incl_rmse_deg'] for result in results]))} for name in BASELINES}}
    report = {key: spec[key] for key in ('checkpoint_sha256', 'training_manifest_sha256', 'selection_sha256', 'sources', 'versions')}
    report.update(spec=spec, spec_sha256=sha256(spec_path), prepared_manifest_sha256=sha256(prepared / 'manifest.json'),
                  trials=results, mean_trial_metrics=means)
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=True)
    for filename, arrays in saved_trials:
        path = output / filename
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
    (output / 'report.json').write_text(encoded)
    print(json.dumps({'output_dir': str(output), 'mean_trial_metrics': means}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prepared-dir', 'spec', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args.checkpoint, args.prepared_dir, args.spec, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'BROAD evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
