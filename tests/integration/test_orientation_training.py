"""THICKENING behaviors:
1. Batched carried/detached TBPTT includes partial batches/chunks and all scored frames.
2. Validation-only selection saves the best eligible epoch and honest family metrics.
3. Seeded training repeats without confirmation reads or validation gradients.
4. Strict warm starts retain normalization and an eligible epoch-zero checkpoint.
"""

from itertools import groupby
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from integration.test_orientation_metrics import poses_z, sha256, write_checkpoint, write_dataset


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')


def fixture(directory, train_angles=None, validation_angles=None):
    directory.mkdir()
    train_angles = train_angles or ([0, 0, .1, .2, .3, .3, .3], [90] * 7, [180] * 7, [45] * 5)
    validation_angles = validation_angles or ([0] * 4, [90] * 3)
    all_entries = []
    for split, angles in (('train', train_angles), ('validation', validation_angles)):
        groups = [('hold' if index % 2 == 0 else 'single_axis', poses_z(values))
                  for index, values in enumerate(angles)]
        manifest = write_dataset(directory / split, groups, split=split)
        for entry in manifest['sequences']:
            entry['path'] = split + '/' + entry['path']
            all_entries.append(entry)
    manifest['sequences'] = all_entries
    manifest['config'].update(train_sequences=len(train_angles), validation_sequences=len(validation_angles))
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def load_arrays(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    arrays = {}
    for entry in manifest['sequences']:
        if entry['split'] == 'confirmation':
            continue
        with np.load(dataset / entry['path'], allow_pickle=False) as saved:
            arrays[entry['sequence_id']] = (entry, saved['sensors'].copy(), saved['quaternion'].copy())
    return arrays


def reference_training(arrays, history, learning_rate=.001, hidden_size=4):
    """Independent optimizer loop; the report exposes trajectory order, not hidden tensors."""
    from model.broad_network import BroadPredictor

    torch.set_num_threads(1)
    torch.manual_seed(23)
    network = BroadPredictor(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    sensors = np.concatenate([values for entry, values, _ in arrays.values() if entry['split'] == 'train'])
    mean, std = sensors.mean(axis=0), np.maximum(sensors.std(axis=0), 1e-8)
    states = []
    for epoch in history:
        order = epoch['trajectory_order']
        for _, identifiers in groupby(order, key=lambda key: len(arrays[key][1])):
            identifiers = list(identifiers)
            for offset in range(0, len(identifiers), 2):
                batch = [arrays[key] for key in identifiers[offset:offset+2]]
                values = np.stack([(item[1] - mean) / std for item in batch])
                targets = np.stack([item[2] for item in batch])
                hidden = None
                for start in range(0, values.shape[1], 3):
                    stop = min(start+3, values.shape[1])
                    prediction, hidden = network(torch.tensor(values[:, start:stop], dtype=torch.float32), hidden)
                    mask = np.arange(start, stop) >= 1
                    if mask.any():
                        prediction = torch.nn.functional.normalize(prediction[:, mask], dim=-1)
                        target = torch.nn.functional.normalize(torch.tensor(targets[:, start:stop][:, mask],
                                                                            dtype=torch.float32), dim=-1)
                        loss = (1 - (prediction * target).sum(dim=-1).clamp(-1, 1).square()).mean()
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
                        optimizer.step()
                    hidden = hidden.detach()
        states.append({name: value.detach().clone() for name, value in network.state_dict().items()})
    return states, {'mean': mean.tolist(), 'std': std.tolist()}


def reference_validation(state, arrays, normalization, hidden_size=4):
    from model.broad_network import BroadPredictor

    network = BroadPredictor(hidden_size=hidden_size)
    network.load_state_dict(state)
    all_errors, families = [], {}
    with torch.no_grad():
        for entry, sensors, targets in arrays.values():
            if entry['split'] != 'validation':
                continue
            values = (sensors - np.asarray(normalization['mean'])) / np.asarray(normalization['std'])
            prediction, _ = network(torch.tensor(values[None], dtype=torch.float32))
            predicted = prediction[0].numpy().astype(np.float64)
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            unit_targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
            errors = np.degrees(2 * np.arccos(np.clip(np.abs((predicted * unit_targets).sum(axis=1)), 0, 1)))[1:]
            all_errors.extend(errors.tolist())
            families.setdefault(entry['family'], []).extend(errors.tolist())
    return float(np.sqrt(np.mean(np.square(all_errors)))), families


class OrientationTrainingTests(unittest.TestCase):
    def cli(self, *arguments, succeeds=True):
        result = subprocess.run([PYTHON, '-m', *arguments], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if succeeds:
            self.assertEqual(result.returncode, 0, 'Training CLI must succeed:\n' + result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def train(self, dataset, output, epochs=1, learning_rate=.001, hidden_size=4, initial=None, succeeds=True):
        arguments = ['model.orientation_train', '--dataset-dir', str(dataset), '--output-dir', str(output),
                     '--epochs', str(epochs), '--hidden-size', str(hidden_size), '--warmup', '1',
                     '--chunk-size', '3', '--batch-size', '2', '--seed', '23', '--learning-rate', str(learning_rate)]
        if initial is not None:
            arguments.extend(['--initial-checkpoint', str(initial)])
        self.cli(*arguments, succeeds=succeeds)
        if succeeds:
            return (json.loads((output / 'report.json').read_text()),
                    torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True))

    def test_batched_carried_detached_chunks_include_partial_batches_and_all_scored_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            fixture(dataset)
            report, checkpoint = self.train(dataset, root / 'trained')
            arrays = load_arrays(dataset)
            order = report['history'][0]['trajectory_order']
            expected_ids = [key for key, (entry, _, _) in arrays.items() if entry['split'] == 'train']
            self.assertCountEqual(order, expected_ids)
            self.assertEqual(len(order), len(expected_ids))
            self.assertEqual((report['optimizer_steps'], report['training_samples_scored']), (8, 22))
            self.assertEqual(report['normalization_origin'], 'train')
            self.assertIsNone(report['initial_checkpoint_sha256'])
            self.assertEqual(checkpoint['config']['batch_size'], 2)
            states, normalization = reference_training(arrays, report['history'])
            np.testing.assert_allclose(checkpoint['normalization']['mean'], normalization['mean'], atol=1e-10)
            np.testing.assert_allclose(checkpoint['normalization']['std'], normalization['std'], atol=1e-10)
            for name, value in checkpoint['state_dict'].items():
                torch.testing.assert_close(value, states[0][name], rtol=1e-6, atol=1e-7)

    def test_saved_checkpoint_is_best_eligible_validation_epoch_with_independent_family_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            fixture(dataset)
            report, checkpoint = self.train(dataset, root / 'trained', epochs=3, learning_rate=.08)
            history = report['history']
            self.assertEqual([entry['epoch'] for entry in history], [1, 2, 3])
            states, normalization = reference_training(load_arrays(dataset), history, learning_rate=.08)
            scores = []
            for entry, state in zip(history, states):
                score, families = reference_validation(state, load_arrays(dataset), normalization)
                scores.append(score)
                self.assertTrue(np.isfinite(entry['training_loss']))
                self.assertAlmostEqual(entry['validation']['full_rmse_deg'], score, delta=1e-4)
                for family, errors in families.items():
                    metrics = entry['validation']['families'][family]
                    self.assertEqual(metrics['sample_count'], len(errors))
                    self.assertAlmostEqual(metrics['full_rmse_deg'], float(np.sqrt(np.mean(np.square(errors)))),
                                           delta=1e-4)
            selected = int(np.argmin(scores)) + 1
            self.assertEqual(report['selected_epoch'], selected)
            self.assertEqual(checkpoint['selected_epoch'], selected)
            self.assertAlmostEqual(report['selected_validation_full_rmse_deg'], min(scores), delta=1e-4)
            self.assertEqual((report['optimizer_steps'], report['training_samples_scored']), (24, 66))
            self.assertEqual((report['selected_optimizer_steps'], report['selected_training_samples_scored']),
                             (selected*8, selected*22))
            for name, value in checkpoint['state_dict'].items():
                torch.testing.assert_close(value, states[selected-1][name], rtol=1e-6, atol=1e-7)
            evaluation = root / 'replay/report.json'
            self.cli('model.orientation_evaluate', '--dataset-dir', str(dataset), '--checkpoint',
                     str(root / 'trained/checkpoint.pt'), '--output', str(evaluation))
            self.assertAlmostEqual(json.loads(evaluation.read_text())['full_rmse_deg'], min(scores), delta=1e-4)

    def test_seeded_training_repeats_and_ignores_confirmation_arrays_and_validation_for_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            manifest = fixture(dataset)
            manifest['sequences'].append({'split': 'confirmation', 'sequence_id': 'confirmation-unopened',
                                         'family': 'hold', 'path': 'never-created.npz', 'sha256': '0' * 64})
            (dataset / 'manifest.json').write_text(json.dumps(manifest))
            first_report, first = self.train(dataset, root / 'first')
            repeated_report, repeated = self.train(dataset, root / 'repeated')
            self.assertEqual(first_report['history'], repeated_report['history'])
            changed = root / 'changed'
            shutil.copytree(dataset, changed)
            changed_manifest = json.loads((changed / 'manifest.json').read_text())
            for entry in changed_manifest['sequences']:
                if entry['split'] == 'validation':
                    path = changed / entry['path']
                    with np.load(path, allow_pickle=False) as saved:
                        sensors, targets = saved['sensors'].copy(), poses_z([135] * len(saved['quaternion']))
                    np.savez_compressed(path, sensors=sensors, quaternion=targets)
                    entry['sha256'] = sha256(path)
            (changed / 'manifest.json').write_text(json.dumps(changed_manifest))
            changed_report, changed_checkpoint = self.train(changed, root / 'changed-trained')
            self.assertEqual(first['normalization'], changed_checkpoint['normalization'])
            self.assertEqual(first_report['history'][0]['training_loss'],
                             changed_report['history'][0]['training_loss'])
            for name, value in first['state_dict'].items():
                torch.testing.assert_close(value, repeated['state_dict'][name], rtol=0, atol=0)
                torch.testing.assert_close(value, changed_checkpoint['state_dict'][name], rtol=0, atol=0)

    def test_warm_start_preserves_normalization_and_epoch_zero_when_best_and_rejects_stale_or_wrong_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset'
            fixture(dataset, train_angles=([90]*7,), validation_angles=([0]*4,))
            initial = root / 'initial.pt'
            write_checkpoint(initial, dataset, std=2.)
            report, checkpoint = self.train(dataset, root / 'trained', hidden_size=96, initial=initial)
            self.assertEqual(report['normalization_origin'], 'initial_checkpoint')
            self.assertEqual(report['initial_checkpoint_sha256'], sha256(initial))
            self.assertEqual((report['selected_epoch'], checkpoint['selected_epoch']), (0, 0))
            self.assertEqual(report['selected_validation_full_rmse_deg'], 0)
            self.assertEqual((report['selected_optimizer_steps'], report['selected_training_samples_scored']), (0, 0))
            self.assertGreater(report['optimizer_steps'], 0)
            self.assertEqual([entry['epoch'] for entry in report['history']], [0, 1])
            original = torch.load(initial, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['normalization'], original['normalization'])
            for name, value in checkpoint['state_dict'].items():
                torch.testing.assert_close(value, original['state_dict'][name], rtol=0, atol=0)
            for reason in ('size', 'stale'):
                with self.subTest(reason=reason):
                    candidate = initial
                    hidden_size = 4
                    if reason == 'stale':
                        stale = torch.load(initial, map_location='cpu', weights_only=True)
                        stale['training_recipe_sources']['model/orientation_train.py'] = '0' * 64
                        candidate = root / 'stale.pt'
                        torch.save(stale, candidate)
                        hidden_size = 96
                    output = root / ('rejected-' + reason)
                    self.train(dataset, output, hidden_size=hidden_size, initial=candidate, succeeds=False)
                    self.assertFalse((output / 'checkpoint.pt').exists())


if __name__ == '__main__':
    unittest.main()
