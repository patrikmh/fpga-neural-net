"""THICKENING: learn and fairly measure future orientation.

Behaviors covered:
1. Forecast loss provides useful sign-invariant angular/norm gradients and masks padding.
2. Chronological training includes partial batches, is repeatable and binds the motion task.
3. Float/integer evaluation replays model outputs and scores fair sensor-only baselines.
4. Sensor-only baselines recover nonidentity poses and advance body-frame angular velocity.
5. Evaluation refuses incompatible tasks, broken provenance and invalid model outputs.
"""

import importlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from model.broad_integer import GoldenBroadPredictor, quantize_inputs, sha256
from model.broad_network import BroadPredictor
from test_orientation_sim import multiply, reference_sensors, unit


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
TASK = {'kind': 'motion_prediction', 'horizon_s': .05, 'model_dt_s': .0035,
        'target': 'absolute_body_to_enu_wxyz'}


class MotionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='motion-training-')
        cls.root = Path(cls.temporary.name)
        cls.data, cls.training, cls.independent = (cls.root / name for name in ('data', 'training', 'independent'))
        cls.run_cli('model.motion_generate', '--output-dir', cls.data, '--seed', '17',
                    '--train-sequences', '3', '--validation-sequences', '2', '--length', '64')
        cls.run_cli('model.motion_generate', '--output-dir', cls.independent, '--seed', '24',
                    '--train-sequences', '0', '--validation-sequences', '2', '--length', '66')
        manifest_path = cls.independent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        entry = manifest['sequences'][1]
        path = cls.independent / entry['path']
        with np.load(path) as saved:
            shorter = {name: saved[name][:-5].copy() for name in saved.files}
        np.savez_compressed(path, **shorter)
        entry['sha256'] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        cls.train_args = ['--dataset-dir', cls.data, '--epochs', '2', '--hidden-size', '96',
                          '--warmup', '4', '--chunk-size', '32', '--batch-size', '2', '--seed', '23',
                          '--learning-rate', '.001']
        cls.run_cli('model.motion_train', *cls.train_args, '--output-dir', cls.training)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def run_cli(cls, module, *arguments, success=True):
        result = subprocess.run([PYTHON, '-m', module, *map(str, arguments)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        if success and result.returncode:
            raise AssertionError(f'{module} must succeed:\n{result.stdout}{result.stderr}')
        return result

    def evaluator(self):
        self.assertIsNotNone(importlib.util.find_spec('model.motion_evaluate'),
                             'Future-pose evaluation must provide model.motion_evaluate')
        return importlib.import_module('model.motion_evaluate')

    def test_forecast_loss_has_sign_invariant_useful_gradients_and_ignores_masked_frames(self):
        module = importlib.import_module('model.motion_train')
        loss_fn = getattr(module, 'forecast_loss', None)
        self.assertTrue(callable(loss_fn), 'Motion training needs forecast_loss with angular and unit-norm supervision')
        raw = torch.tensor([[[.001, 1., 0., 0.], [2., 0., 0., 0.], [.3, .4, .5, .6]]], requires_grad=True)
        targets = torch.tensor([[[1., 0., 0., 0.]] * 3])
        mask = torch.tensor([[True, True, False]])
        loss = loss_fn(raw, targets, mask)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertAlmostEqual(float(loss.detach()), float(loss_fn(-raw, targets, mask).detach()), places=7)
        loss.backward()
        self.assertLess(float(raw.grad[0, 0, 0]), -.1,
                        'Near-180-degree errors need a useful gradient toward the target component')
        self.assertGreater(float(raw.grad[0, 1, 0]), 0, 'Oversized output norm must be pulled toward one')
        torch.testing.assert_close(raw.grad[0, 2], torch.zeros(4), rtol=0, atol=0)
        for invalid in (torch.zeros((1, 1, 4)), torch.full((1, 1, 4), float('nan'))):
            with self.subTest(raw=invalid.tolist()), self.assertRaises(ValueError):
                loss_fn(invalid, targets[:, :1], mask[:, :1])

    def test_training_scores_partial_batches_repeats_and_binds_best_validation_to_task(self):
        report = json.loads((self.training / 'report.json').read_text())
        checkpoint = torch.load(self.training / 'checkpoint.pt', weights_only=True)
        self.assertEqual(checkpoint['config']['task'], TASK)
        self.assertEqual(report['training_samples_scored'], 2 * 3 * (49 - 4))
        self.assertEqual(report['optimizer_steps'], 8)
        best = min(report['history'], key=lambda epoch: epoch['validation']['full_rmse_deg'])
        self.assertEqual(report['selected_epoch'], best['epoch'])
        self.assertEqual(best['validation']['sample_count'], 2 * (49 - 4))
        manifest = json.loads((self.data / 'manifest.json').read_text())
        sensors = []
        for entry in manifest['sequences']:
            if entry['split'] == 'train':
                with np.load(self.data / entry['path']) as saved:
                    sensors.append(saved['sensors'])
        sensors = np.concatenate(sensors)
        np.testing.assert_allclose(checkpoint['normalization']['mean'], sensors.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(checkpoint['normalization']['std'], np.maximum(sensors.std(axis=0), 1e-8), atol=1e-12)
        self.assertIn('model/motion_evaluate.py', checkpoint['training_recipe_sources'],
                      'Prediction recipe must bind its evaluator source')
        repeated = self.root / 'repeated'
        self.run_cli('model.motion_train', *self.train_args, '--output-dir', repeated)
        duplicate = torch.load(repeated / 'checkpoint.pt', weights_only=True)
        for name, value in checkpoint['state_dict'].items():
            torch.testing.assert_close(duplicate['state_dict'][name], value, rtol=0, atol=0)
        mismatched = dict(checkpoint, config={**checkpoint['config'], 'task': {**TASK, 'horizon_s': .1}})
        mismatch_path = self.root / 'different-horizon.pt'
        torch.save(mismatched, mismatch_path)
        result = self.run_cli('model.motion_train', *self.train_args, '--initial-checkpoint', mismatch_path,
                              '--output-dir', self.root / 'rejected-initial', success=False)
        self.assertNotEqual(result.returncode, 0, 'Warm-starting a different prediction horizon must be rejected')

    def test_evaluation_replays_full_model_and_fair_baselines_on_independent_data(self):
        evaluator = self.evaluator()
        export = self.root / 'export'
        self.run_cli('model.broad_export_integer', '--checkpoint', self.training / 'checkpoint.pt', '--output-dir', export)
        checkpoint = torch.load(self.training / 'checkpoint.pt', weights_only=True)
        mean, std = (np.asarray(checkpoint['normalization'][key]) for key in ('mean', 'std'))
        entries = json.loads((self.independent / 'manifest.json').read_text())['sequences']
        for mode in ('float', 'integer'):
            with self.subTest(mode=mode):
                output = self.root / f'evaluation-{mode}'
                source = ['--checkpoint', self.training / 'checkpoint.pt'] if mode == 'float' else ['--export-dir', export]
                self.run_cli('model.motion_evaluate', '--dataset-dir', self.independent, '--split', 'validation',
                              *source, '--output', output / 'report.json')
                report = json.loads((output / 'report.json').read_text())
                self.assertEqual(report['task'], TASK)
                self.assertEqual(report['sample_count'], (51 - 4) + (46 - 4))
                self.assertEqual(report['cold_start']['sample_count'], 8)
                self.assertNotEqual(report['dataset_manifest_sha256'], report['training_dataset_manifest_sha256'])
                predictor = GoldenBroadPredictor.from_export(export)
                counters = dict(input_clipping_count=0, lut_clamp_count=0, state_clipping_count=0, raw_clipping_count=0)
                all_errors = []
                moving_count = stationary_count = 0
                baseline_errors = {'sensor_hold': [], 'sensor_constant_velocity': []}
                network = BroadPredictor(hidden_size=96)
                network.load_state_dict(checkpoint['state_dict'])
                network.eval()
                for entry, sequence_report in zip(entries, report['sequences']):
                    with np.load(self.independent / entry['path']) as original:
                        sensors, targets = original['sensors'], original['quaternion']
                    with np.load(output / sequence_report['predictions_file']) as saved:
                        normalized = (sensors - mean) / std
                        if mode == 'integer':
                            inputs, clipped = quantize_inputs(normalized)
                            counters['input_clipping_count'] += clipped
                            np.testing.assert_array_equal(saved['quantized_inputs'], inputs)
                            effective = inputs.astype(np.float64) / 2048 * std + mean
                            predictor.reset()
                            replay = [predictor.step(row) for row in inputs]
                            hidden, raw = np.asarray([row[0] for row in replay]), np.asarray([row[1] for row in replay])
                            for key, value in predictor.counters.items():
                                counters[key] += value
                        else:
                            inputs = normalized.astype(np.float32)
                            effective = inputs.astype(np.float64) * std + mean
                            state = torch.zeros((1, 96))
                            hidden, raw = [], []
                            with torch.no_grad():
                                for row in inputs:
                                    values, state = network.cell(torch.tensor(row)[None], state, 1.)
                                    hidden.append(state[0].numpy().copy())
                                    raw.append(network.readout(values)[0].numpy().copy())
                            hidden, raw = np.asarray(hidden), np.asarray(raw)
                        np.testing.assert_array_equal(saved['model_inputs'], inputs)
                        np.testing.assert_allclose(saved['effective_sensors'], effective, atol=1e-12)
                        np.testing.assert_array_equal(saved['hidden'], hidden)
                        np.testing.assert_array_equal(saved['raw_outputs'], raw)
                        predictions = raw.astype(np.float64) / np.linalg.norm(raw, axis=1, keepdims=True)
                        np.testing.assert_allclose(saved['predictions'], predictions, atol=1e-12)
                        np.testing.assert_array_equal(saved['targets'], targets)
                        errors = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(predictions * targets, axis=1)), 0, 1)))
                        np.testing.assert_allclose(saved['angle_errors_deg'], errors, atol=1e-6)
                        np.testing.assert_array_equal(saved['scored_mask'], np.arange(len(targets)) >= 4)
                        moving = np.linalg.norm(sensors[:, 3:6], axis=1) > 1e-8
                        np.testing.assert_array_equal(saved['moving_mask'], moving,
                                                      err_msg='Physical motion categories must ignore model-input quantization noise')
                        moving_count += int(np.count_nonzero(moving[4:]))
                        stationary_count += int(np.count_nonzero(~moving[4:]))
                        all_errors.extend(errors[4:])
                        baselines = evaluator.sensor_baselines(effective, .05)
                        for name, values in baselines.items():
                            np.testing.assert_allclose(saved[name + '_predictions'], values, atol=1e-12)
                            baseline_error = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(values * targets, axis=1)), 0, 1)))
                            np.testing.assert_allclose(saved[name + '_errors_deg'], baseline_error, atol=1e-6)
                            baseline_errors[name].extend(baseline_error[4:])
                self.assertAlmostEqual(report['full_rmse_deg'], float(np.sqrt(np.mean(np.square(all_errors)))), places=6)
                self.assertEqual(sum(family['sample_count'] for family in report['families'].values()), report['sample_count'])
                self.assertEqual(report['moving']['sample_count'], moving_count)
                self.assertEqual(report['stationary']['sample_count'], stationary_count)
                for name, errors in baseline_errors.items():
                    self.assertEqual(report['baselines'][name]['sample_count'], report['sample_count'])
                    self.assertAlmostEqual(report['baselines'][name]['full_rmse_deg'], float(np.sqrt(np.mean(np.square(errors)))), places=6)
                if mode == 'integer':
                    for name, value in counters.items():
                        self.assertEqual(report[name], value)
        altered = self.root / 'altered-current-annotations'
        shutil.copytree(self.independent, altered)
        manifest = json.loads((altered / 'manifest.json').read_text())
        for entry in manifest['sequences']:
            path = altered / entry['path']
            with np.load(path) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays['current_quaternion'] = np.tile([1., 0., 0., 0.], (len(arrays['sensors']), 1))
            np.savez_compressed(path, **arrays)
            entry['sha256'] = sha256(path)
        (altered / 'manifest.json').write_text(json.dumps(manifest))
        altered_output = self.root / 'evaluation-altered-current'
        self.run_cli('model.motion_evaluate', '--dataset-dir', altered, '--export-dir', export,
                      '--output', altered_output / 'report.json')
        altered_report = json.loads((altered_output / 'report.json').read_text())
        for original, changed in zip(report['sequences'], altered_report['sequences']):
            with np.load(output / original['predictions_file']) as before, np.load(altered_output / changed['predictions_file']) as after:
                for name in ('sensor_hold_predictions', 'sensor_constant_velocity_predictions'):
                    np.testing.assert_array_equal(before[name], after[name],
                                                  err_msg='Sensor baselines must not read current-pose annotations')

    def test_sensor_baselines_recover_nonidentity_pose_and_right_multiply_body_rotation(self):
        baselines_fn = self.evaluator().sensor_baselines
        initial = unit([.4, -.2, .7, .3])
        axis = unit([.3, -.4, .5])
        omega = axis * 2.3
        poses = np.stack([multiply(initial, np.r_[math.cos(2.3 * t / 2), axis * math.sin(2.3 * t / 2)])
                          for t in np.arange(1, 33) * .0035])
        sensors = np.stack([reference_sensors(initial if i == 0 else poses[i-1], pose)
                            for i, pose in enumerate(poses)])
        predictions = baselines_fn(sensors, .05)
        self.assertEqual(set(predictions), {'sensor_hold', 'sensor_constant_velocity'})
        future_delta = np.r_[math.cos(np.linalg.norm(omega) * .05 / 2), axis * math.sin(np.linalg.norm(omega) * .05 / 2)]
        future = np.stack([multiply(pose, future_delta) for pose in poses])
        for name, expected in (('sensor_hold', poses), ('sensor_constant_velocity', future)):
            np.testing.assert_allclose(np.abs(np.sum(predictions[name] * expected, axis=1)), 1, atol=1e-11)

    def test_evaluation_rejects_wrong_task_source_invalid_outputs_and_occupied_destination(self):
        self.evaluator()
        checkpoint = torch.load(self.training / 'checkpoint.pt', weights_only=True)
        cases = ('task', 'horizon', 'source', 'zero', 'nonfinite', 'occupied')
        for case in cases:
            with self.subTest(case=case):
                candidate = {**checkpoint, 'config': {**checkpoint['config']},
                             'state_dict': {name: value.clone() for name, value in checkpoint['state_dict'].items()},
                             'training_recipe_sources': dict(checkpoint['training_recipe_sources'])}
                if case == 'task':
                    candidate['config']['task'] = {**TASK, 'kind': 'orientation_estimation'}
                elif case == 'horizon':
                    candidate['config']['task'] = {**TASK, 'horizon_s': .1}
                elif case == 'source':
                    candidate['training_recipe_sources']['model/motion_train.py'] = '0' * 64
                elif case in ('zero', 'nonfinite'):
                    candidate['state_dict']['readout.weight'].zero_()
                    candidate['state_dict']['readout.bias'].fill_(0 if case == 'zero' else float('nan'))
                path = self.root / f'guard-{case}.pt'
                torch.save(candidate, path)
                output = self.root / f'rejected-evaluation-{case}'
                if case == 'occupied':
                    output.mkdir()
                    (output / 'keep.txt').write_text('preserve')
                result = self.run_cli('model.motion_evaluate', '--dataset-dir', self.independent,
                                      '--checkpoint', path, '--output', output / 'report.json', success=False)
                self.assertNotEqual(result.returncode, 0, f'Evaluator must reject {case}')
                self.assertNotIn('No module named', result.stderr)
                self.assertFalse((output / 'report.json').exists(), 'Rejected evaluations must not claim metrics')
                if case == 'occupied':
                    self.assertEqual((output / 'keep.txt').read_text(), 'preserve')
