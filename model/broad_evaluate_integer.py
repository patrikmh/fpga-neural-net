"""Continuous validation of the BROAD integer model; never reads held-out trials."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.broad_evaluate import orientation_metrics
from model.broad_integer import GoldenBroadPredictor, artifact_sources, quantize_inputs, runtime_versions, sha256
from model.broad_network import BroadPredictor
from model.broad_train import load_trials, predict_trial


COUNTERS = ('input_clipping_count', 'lut_clamp_count', 'state_clipping_count', 'raw_clipping_count', 'zero_quaternion_count')


def evaluate(export_dir, prepared, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    predictor = GoldenBroadPredictor.from_export(export_dir)
    artifact, checkpoint = predictor.manifest, predictor.checkpoint
    export_hashes = {str(path.relative_to(export_dir)): sha256(path) for path in export_dir.rglob('*') if path.is_file()}
    prepared_hash = sha256(prepared / 'manifest.json')
    if prepared_hash != checkpoint['prepared_manifest_sha256']:
        raise ValueError('Prepared training/validation manifest SHA256 identity mismatch')
    trials = [trial for trial in load_trials(prepared) if trial['split'] == 'validation']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    network = BroadPredictor(hidden_size=artifact['config']['hidden_size']).eval()
    network.load_state_dict(checkpoint['state_dict'])
    mean, std = (np.asarray(artifact['normalization'][name], dtype=np.float64) for name in ('mean', 'std'))
    warmup = artifact['config']['warmup']
    totals = {name: 0 for name in COUNTERS}
    results, arrays_to_save = [], []
    for trial in trials:
        sensors, targets = trial['sensors'], trial['quaternion']
        quantized, input_clips = quantize_inputs((sensors.astype(np.float64) - mean) / std)
        hidden = np.empty((len(sensors), predictor.hidden_size), dtype=np.int16)
        raw = np.empty((len(sensors), 4), dtype=np.int16)
        predictor.reset()
        for index, values in enumerate(quantized):
            hidden[index], raw[index] = predictor.step(values)
        values = raw.astype(np.float64) / 8192
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        zero = norms[:, 0] == 0
        integer_predictions = values / np.where(norms > 0, norms, 1)
        integer_predictions[zero] = [1, 0, 0, 0]
        floating = predict_trial(network, sensors, artifact['normalization'])
        mask = (trial['movement'] & trial['reference_valid']).copy()
        mask[:warmup] = False
        metrics = {name: orientation_metrics(prediction, targets, mask, 1e-12)
                   for name, prediction in (('float', floating), ('integer', integer_predictions))}
        counters = dict(predictor.counters, input_clipping_count=input_clips, zero_quaternion_count=int(zero.sum()))
        for name in COUNTERS:
            totals[name] += counters[name]
        stem = f"validation/{trial['trial_id']:02d}"
        results.append(dict(trial_id=trial['trial_id'], predictions_file=stem + '.npz', vectors_file=stem + '.hex',
                            **metrics, **counters))
        arrays_to_save.append((stem, dict(sensors=sensors, targets=targets, quantized_inputs=quantized,
            hidden=hidden, raw_outputs=raw, integer_predictions=integer_predictions, float_predictions=floating,
            movement=trial['movement'], reference_valid=trial['reference_valid'], sample_index=trial['sample_index'])))
        print(f"trial {trial['trial_id']}: float/integer full RMSE "
              f"{metrics['float']['full_rmse_deg']:.6f}/{metrics['integer']['full_rmse_deg']:.6f} deg", flush=True)
    means = {}
    for kind in ('float', 'integer'):
        means[kind] = {}
        for name in ('full_rmse_deg', 'inclination_rmse_deg', 'heading_rmse_deg'):
            values = [entry[kind][name] for entry in results if entry[kind][name] is not None]
            means[kind][name] = float(np.mean(values)) if values else None
    if not results:
        raise ValueError('No validation trials')
    zero_keys = ('input_clipping_count', 'state_clipping_count', 'raw_clipping_count', 'zero_quaternion_count')
    passed = means['integer']['full_rmse_deg'] <= means['float']['full_rmse_deg'] + .25 and all(totals[name] == 0 for name in zero_keys)
    if export_hashes != {str(path.relative_to(export_dir)): sha256(path) for path in export_dir.rglob('*') if path.is_file()}:
        raise ValueError('Integer export identity changed during evaluation')
    if prepared_hash != sha256(prepared / 'manifest.json') or artifact['sources'] != artifact_sources(checkpoint):
        raise ValueError('Prepared manifest or runtime source identity changed during evaluation')
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for stem, arrays in arrays_to_save:
        path = output / (stem + '.npz')
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        outputs[stem + '.npz'] = sha256(path)
        vectors = output / (stem + '.hex')
        with vectors.open('w') as stream:
            for inputs, hidden, raw in zip(arrays['quantized_inputs'], arrays['hidden'], arrays['raw_outputs']):
                stream.write(''.join(f'{int(value) & 65535:04x}' for value in np.concatenate((inputs, hidden, raw))) + '\n')
        outputs[stem + '.hex'] = sha256(vectors)
    report = {'split': 'validation', 'warmup': warmup, 'trials': results, 'mean_trial_metrics': means,
              'acceptance': {'max_full_rmse_increase_deg': .25, 'required_zero_counts': list(zero_keys)},
              'meets_precision_target': passed, **totals, 'outputs': outputs,
              'export_manifest_sha256': export_hashes['manifest.json'],
              'checkpoint_sha256': artifact['checkpoint_sha256'], 'prepared_manifest_sha256': prepared_hash,
              'sources': artifact['sources'], 'versions': runtime_versions(),
              'vector_order': 'nine inputs, H new hidden states, four raw outputs; signed16 MSB-first words'}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'meets_precision_target': passed}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir', 'prepared-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args.export_dir, args.prepared_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'BROAD integer evaluation failed: {error}\n')


if __name__ == '__main__':
    main()
