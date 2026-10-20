"""Score source-bound float or integer CfC predictions on independent trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.broad_integer import (GoldenBroadPredictor, artifact_sources, quantize_inputs,
                                sha256, validate_checkpoint)
from model.broad_network import BroadPredictor
from model.orientation_generate import load_sequences


def metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    result = {'sample_count': len(errors), 'above_5_deg_count': int(np.count_nonzero(errors > 5))}
    result.update(dict.fromkeys(('full_rmse_deg', 'p50_deg', 'p95_deg', 'max_deg')))
    if len(errors):
        result.update(full_rmse_deg=float(np.sqrt(np.mean(errors ** 2))),
                      p50_deg=float(np.percentile(errors, 50)),
                      p95_deg=float(np.percentile(errors, 95)), max_deg=float(errors.max()))
    return result


def predict_float_sequence(network, sensors, normalization):
    mean, std = (np.asarray(normalization[name], dtype=np.float64) for name in ('mean', 'std'))
    inputs = torch.tensor((sensors.astype(np.float64) - mean) / std, dtype=torch.float32)
    hidden = torch.zeros((1, network.hidden_size), dtype=inputs.dtype)
    states, outputs = [], []
    network.eval()
    with torch.no_grad():
        for value in inputs:
            output, hidden = network.cell(value[None], hidden, 1.)
            raw = network.readout(output)
            if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(raw).all()):
                raise ValueError('Nonfinite float hidden state or output quaternion')
            states.append(hidden[0].numpy().copy())
            outputs.append(raw[0].numpy().copy())
    return np.asarray(states), np.asarray(outputs)


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
    normalization = checkpoint['normalization']
    warmup = checkpoint['config']['warmup']
    mean, std = (np.asarray(normalization[name], dtype=np.float64) for name in ('mean', 'std'))
    sequences = load_sequences(args.dataset_dir, args.split)
    output.mkdir(parents=True, exist_ok=True)
    results, all_errors, cold_errors, moving_errors, stationary_errors = [], [], [], [], []
    families = {}
    total_count, clipping_count = 0, 0
    for number, (entry, sensors, targets) in enumerate(sequences):
        arrays = {}
        if mode == 'float':
            hidden, raw = predict_float_sequence(network, sensors, normalization)
        else:
            quantized, clipped = quantize_inputs((sensors.astype(np.float64) - mean) / std)
            clipping_count += clipped
            predictor.reset()
            states, outputs = [], []
            for inputs in quantized:
                state, prediction = predictor.step(inputs)
                states.append(state)
                outputs.append(prediction)
            hidden, raw = np.asarray(states, dtype=np.int16), np.asarray(outputs, dtype=np.int16)
            arrays['quantized_inputs'] = quantized
        if not np.isfinite(hidden).all() or not np.isfinite(raw).all():
            raise ValueError('Nonfinite hidden state or output quaternion')
        norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError('Model emitted a zero or nonfinite output quaternion')
        predictions = raw.astype(np.float64) / norms
        unit_targets = targets.astype(np.float64)
        unit_targets /= np.linalg.norm(unit_targets, axis=1, keepdims=True)
        errors = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(
            predictions * unit_targets, axis=1)), 0, 1)))
        scored = np.arange(len(errors)) >= warmup
        moving = np.zeros(len(errors), dtype=bool)
        moving[1:] = np.abs(np.sum(unit_targets[:-1] * unit_targets[1:], axis=1)) < 1 - 1e-12
        filename = f'predictions-{number:04d}.npz'
        np.savez_compressed(output / filename, **arrays, hidden=hidden,
                            raw_outputs=raw, targets=targets, predictions=predictions,
                            angle_errors_deg=errors, scored_mask=scored, moving_mask=moving)
        result = metrics(errors[scored])
        result.update(sequence_id=entry['sequence_id'], family=entry['family'],
                      total_sample_count=len(errors), predictions_file=filename,
                      cold_start=metrics(errors[~scored]), stationary=metrics(errors[scored & ~moving]),
                      moving=metrics(errors[scored & moving]))
        results.append(result)
        all_errors.extend(errors[scored].tolist())
        cold_errors.extend(errors[~scored].tolist())
        moving_errors.extend(errors[scored & moving].tolist())
        stationary_errors.extend(errors[scored & ~moving].tolist())
        families.setdefault(entry['family'], []).extend(errors[scored].tolist())
        total_count += len(errors)
    if not all_errors:
        raise ValueError('No scored samples remain after per-sequence warmup')
    report = metrics(all_errors)
    report.update(split=args.split, mode=mode, total_sample_count=total_count, warmup_samples=warmup,
                  sequences=results, cold_start=metrics(cold_errors), stationary=metrics(stationary_errors),
                  moving=metrics(moving_errors), families={name: metrics(values) for name, values in families.items()},
                  input_clipping_count=clipping_count if mode == 'integer' else None,
                  dataset_manifest_sha256=sha256(args.dataset_dir / 'manifest.json'),
                  training_dataset_manifest_sha256=checkpoint['dataset_manifest_sha256'],
                  checkpoint_sha256=checkpoint_hash,
                  sources={**sources, 'model/orientation_evaluate.py': sha256(Path(__file__))})
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
        parser.exit(1, f'Orientation evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
