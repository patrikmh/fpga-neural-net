"""Replay motion forecasts and sensor-only baselines on identical effective inputs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from host.orientation_sim import MAGNETIC_DIRECTION
from model.broad_integer import (GoldenBroadPredictor, artifact_sources, quantize_inputs,
                                sha256, validate_checkpoint)
from model.broad_network import BroadPredictor
from model.motion_generate import load_sequences
from model.orientation_evaluate import metrics, predict_float_sequence


def matrix_quaternion(matrix):
    """Convert a body-to-Earth rotation matrix, retaining stable half-turn cases."""
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2 * math.sqrt(1 + trace)
        result = [scale / 4, (matrix[2, 1] - matrix[1, 2]) / scale,
                  (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale]
    else:
        index = int(np.argmax(np.diag(matrix)))
        other, last = (index + 1) % 3, (index + 2) % 3
        scale = 2 * math.sqrt(max(0., 1 + matrix[index, index] - matrix[other, other] - matrix[last, last]))
        result = np.zeros(4)
        result[0] = (matrix[last, other] - matrix[other, last]) / scale
        result[index + 1] = scale / 4
        result[other + 1] = (matrix[other, index] + matrix[index, other]) / scale
        result[last + 1] = (matrix[last, index] + matrix[index, last]) / scale
    result = np.asarray(result)
    return result / np.linalg.norm(result)


def vector_basis(gravity, magnetic):
    first = gravity / np.linalg.norm(gravity)
    second = np.cross(first, magnetic)
    norm = np.linalg.norm(second)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('Sensor baseline requires noncollinear finite gravity and magnetic vectors')
    second /= norm
    return np.column_stack((first, second, np.cross(first, second)))


def sensor_baselines(sensors, horizon_s):
    sensors = np.asarray(sensors, dtype=np.float64)
    if (sensors.ndim != 2 or sensors.shape[1] != 9 or not np.isfinite(sensors).all()
            or not math.isfinite(horizon_s) or horizon_s <= 0):
        raise ValueError('Baseline requires finite nine-channel sensors and a positive horizon')
    earth = vector_basis(np.array([0., 0., 1.]), np.asarray(MAGNETIC_DIRECTION))
    held, extrapolated = [], []
    for row in sensors:
        pose = matrix_quaternion(earth @ vector_basis(row[:3], row[6:9]).T)
        speed = np.linalg.norm(row[3:6])
        angle = speed * horizon_s / 2
        delta = np.r_[math.cos(angle), row[3:6] * (math.sin(angle) / speed if speed else 0.)]
        w, x, y, z = pose
        a, b, c, d = delta
        future = np.array([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                           w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])
        held.append(pose)
        extrapolated.append(future / np.linalg.norm(future))
    return {'sensor_hold': np.asarray(held), 'sensor_constant_velocity': np.asarray(extrapolated)}


def angle_errors(predictions, targets):
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(predictions * targets, axis=1)), 0, 1)))


def evaluate(args):
    output = args.output.parent
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    torch.set_num_threads(1)
    mode = 'float' if args.checkpoint else 'integer'
    if mode == 'float':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        size = validate_checkpoint(checkpoint)
        sources = artifact_sources(checkpoint)
        network = BroadPredictor(hidden_size=size)
        network.load_state_dict(checkpoint['state_dict'])
        checkpoint_hash = sha256(args.checkpoint)
    else:
        predictor = GoldenBroadPredictor.from_export(args.export_dir)
        checkpoint, manifest = predictor.checkpoint, predictor.manifest
        sources = manifest['sources']
        checkpoint_hash = manifest['checkpoint_sha256']
    sequences = load_sequences(args.dataset_dir, args.split)
    dataset_manifest = json.loads((args.dataset_dir / 'manifest.json').read_text())
    task = dataset_manifest['task']
    if checkpoint['config'].get('task') != task:
        raise ValueError('Checkpoint and dataset motion task/horizon mismatch')
    warmup = checkpoint['config']['warmup']
    if not any(len(sensors) > warmup for _, sensors, _ in sequences):
        raise ValueError('No scored samples remain after per-sequence warmup')
    mean, std = (np.asarray(checkpoint['normalization'][name], dtype=np.float64) for name in ('mean', 'std'))
    output.mkdir(parents=True, exist_ok=True)
    results, all_errors, cold_errors, moving_errors, stationary_errors = [], [], [], [], []
    families, baseline_errors, baseline_cold, baseline_families = {}, {}, {}, {}
    counters = dict(input_clipping_count=0, lut_clamp_count=0, state_clipping_count=0, raw_clipping_count=0)
    total = 0
    for number, (entry, sensors, targets) in enumerate(sequences):
        arrays = {}
        normalized = (sensors.astype(np.float64) - mean) / std
        if mode == 'float':
            inputs = normalized.astype(np.float32)
            effective = inputs.astype(np.float64) * std + mean
            hidden, raw = predict_float_sequence(network, sensors, checkpoint['normalization'])
        else:
            inputs, clipped = quantize_inputs(normalized)
            counters['input_clipping_count'] += clipped
            effective = inputs.astype(np.float64) / 2048 * std + mean
            predictor.reset()
            replay = [predictor.step(row) for row in inputs]
            hidden, raw = np.asarray([row[0] for row in replay]), np.asarray([row[1] for row in replay])
            for name, count in predictor.counters.items():
                counters[name] += count
            arrays['quantized_inputs'] = inputs
        norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
        if not np.isfinite(hidden).all() or not np.isfinite(raw).all() or np.any(norms == 0):
            raise ValueError('Model emitted a zero or nonfinite state/output quaternion')
        predictions = raw.astype(np.float64) / norms
        errors = angle_errors(predictions, targets)
        scored = np.arange(len(errors)) >= warmup
        moving = np.linalg.norm(sensors[:, 3:6], axis=1) > 1e-8
        per_baseline = {}
        for name, values in sensor_baselines(effective, task['horizon_s']).items():
            baseline_error = angle_errors(values, targets)
            arrays[name + '_predictions'] = values
            arrays[name + '_errors_deg'] = baseline_error
            baseline_errors.setdefault(name, []).extend(baseline_error[scored].tolist())
            baseline_cold.setdefault(name, []).extend(baseline_error[~scored].tolist())
            baseline_families.setdefault(name, {}).setdefault(entry['family'], []).extend(baseline_error[scored].tolist())
            per_baseline[name] = metrics(baseline_error[scored])
        filename = f'predictions-{number:04d}.npz'
        np.savez_compressed(output / filename, **arrays, model_inputs=inputs, effective_sensors=effective,
                            hidden=hidden, raw_outputs=raw, targets=targets, predictions=predictions,
                            angle_errors_deg=errors, scored_mask=scored, moving_mask=moving)
        result = metrics(errors[scored])
        result.update(sequence_id=entry['sequence_id'], family=entry['family'],
                      total_sample_count=len(errors), predictions_file=filename,
                      cold_start=metrics(errors[~scored]), baselines=per_baseline)
        results.append(result)
        all_errors.extend(errors[scored].tolist())
        cold_errors.extend(errors[~scored].tolist())
        moving_errors.extend(errors[scored & moving].tolist())
        stationary_errors.extend(errors[scored & ~moving].tolist())
        families.setdefault(entry['family'], []).extend(errors[scored].tolist())
        total += len(errors)
    report = metrics(all_errors)
    report.update(task=task, split=args.split, mode=mode, total_sample_count=total, warmup_samples=warmup,
                  sequences=results, cold_start=metrics(cold_errors), moving=metrics(moving_errors),
                  stationary=metrics(stationary_errors), families={name: metrics(values) for name, values in families.items()},
                  baselines={name: {**metrics(values), 'cold_start': metrics(baseline_cold[name]),
                                    'families': {family: metrics(errors) for family, errors in baseline_families[name].items()}}
                             for name, values in baseline_errors.items()},
                  **{name: count if mode == 'integer' else None for name, count in counters.items()},
                  dataset_manifest_sha256=sha256(args.dataset_dir / 'manifest.json'),
                  training_dataset_manifest_sha256=checkpoint['dataset_manifest_sha256'],
                  checkpoint_sha256=checkpoint_hash,
                  sources={**sources, 'model/motion_evaluate.py': sha256(Path(__file__))})
    if mode == 'integer':
        report['export_manifest_sha256'] = sha256(args.export_dir / 'manifest.json')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'confirmation'), default='validation')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--export-dir', type=Path)
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Motion evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
