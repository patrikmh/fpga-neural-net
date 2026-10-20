"""Replay virtual-body validation using the strict integer CfC implementation."""

import argparse
import json
from pathlib import Path

import numpy as np

from model.broad_integer import GoldenBroadPredictor, quantize_inputs
from model.orientation_generate import load_sequences


def evaluate(args):
    output = args.output.parent
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    predictor = GoldenBroadPredictor.from_export(args.export_dir)
    manifest = json.loads((args.export_dir / 'manifest.json').read_text())
    mean, std = (np.asarray(manifest['normalization'][name], dtype=np.float64)
                 for name in ('mean', 'std'))
    sequences = load_sequences(args.dataset_dir, args.split)
    output.mkdir(parents=True, exist_ok=True)
    results, all_errors = [], []
    for number, (entry, sensors, targets) in enumerate(sequences):
        quantized, _ = quantize_inputs((sensors.astype(np.float64) - mean) / std)
        predictor.reset()
        hidden, raw = [], []
        for inputs in quantized:
            state, prediction = predictor.step(inputs)
            hidden.append(state.copy())
            raw.append(prediction.copy())
        hidden, raw = np.asarray(hidden, dtype=np.int16), np.asarray(raw, dtype=np.int16)
        norms = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Integer model emitted a zero output quaternion')
        predictions = raw.astype(np.float64) / norms
        unit_targets = targets.astype(np.float64)
        unit_targets /= np.linalg.norm(unit_targets, axis=1, keepdims=True)
        errors = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(
            predictions * unit_targets, axis=1)), 0, 1)))
        filename = f'predictions-{number:04d}.npz'
        np.savez_compressed(output / filename, quantized_inputs=quantized, hidden=hidden,
                            raw_outputs=raw, targets=targets, predictions=predictions,
                            angle_errors_deg=errors)
        results.append({'sequence_id': entry['sequence_id'], 'sample_count': len(errors),
                        'full_rmse_deg': float(np.sqrt(np.mean(errors ** 2))),
                        'predictions_file': filename})
        all_errors.extend(errors.tolist())
    report = {'split': args.split, 'sample_count': len(all_errors), 'sequences': results,
              'full_rmse_deg': float(np.sqrt(np.mean(np.square(all_errors))))}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Orientation evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
