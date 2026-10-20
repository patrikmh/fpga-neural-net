"""THICKENING: independently scored orientation evaluation through real CLIs.

Behaviors covered:
1. Float/integer reports score every sequence with sample-weighted, sign-invariant
   full-quaternion metrics, retaining warmup samples separately.
2. Motion masks use true pose changes; clipping is counted and zero/empty results fail.
3. Independent confirmation data preserves source/checkpoint/dataset provenance.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from integration.test_model_integer import half_away, saturate
from test_orientation_sim import reference_sensors


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
RECIPE = ('model/orientation_generate.py', 'model/orientation_train.py',
          'model/orientation_evaluate.py', 'host/orientation_sim.py')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def poses_z(angles):
    radians = np.radians(np.asarray(angles)) / 2
    return np.column_stack((np.cos(radians), np.zeros((len(radians), 2)), np.sin(radians)))


def write_dataset(directory, sequences, split='validation'):
    directory.mkdir()
    entries = []
    for index, (family, poses) in enumerate(sequences):
        sensors = np.stack([reference_sensors(poses[max(0, step-1)], pose)
                            for step, pose in enumerate(poses)])
        sequence_id = f'{split}-{index:04d}'
        path = directory / (sequence_id + '.npz')
        np.savez_compressed(path, sensors=sensors, quaternion=poses)
        entries.append({'split': split, 'sequence_id': sequence_id, 'family': family,
                        'path': path.name, 'sha256': sha256(path)})
    manifest = {'dt_s': .0035, 'sensor_source': 'virtual_body', 'sequences': entries,
                'max_angular_rate_rad_s': 6, 'quaternion_convention': 'wxyz_body_to_ENU',
                'sensor_order': [f'{sensor}_{axis}' for sensor in ('acc', 'gyr', 'mag') for axis in 'xyz'],
                'sensor_units': {'acc': 'm/s^2', 'gyr': 'rad/s', 'mag': 'uT'},
                'sources': {name: sha256(ROOT / name) for name in
                            ('model/orientation_generate.py', 'host/orientation_sim.py')},
                'config': {'families': sorted({family for family, _ in sequences}),
                           'train_sequences': len(entries) if split == 'train' else 0,
                           'validation_sequences': len(entries) if split == 'validation' else 0,
                           'confirmation_sequences': len(entries) if split == 'confirmation' else 0}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


def write_checkpoint(path, dataset, zero=False, std=1.):
    from model.broad_network import BroadPredictor
    from model.broad_train_stateful import current_provenance

    network = BroadPredictor(hidden_size=96)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        if not zero:
            network.readout.bias[0] = 1
    checkpoint = {'state_dict': network.state_dict(),
                  'config': {'input_size': 9, 'hidden_size': 96, 'output_size': 4,
                             'mode': 'default', 'backbone_layers': 0, 'dt': 1., 'warmup': 1},
                  'normalization': {'mean': [0.] * 9, 'std': [std] * 9},
                  'provenance': current_provenance(),
                  'training_recipe_sources': {name: sha256(ROOT / name) for name in RECIPE},
                  'dataset_manifest_sha256': sha256(dataset / 'manifest.json')}
    torch.save(checkpoint, path)


def errors_for_identity(poses):
    poses = np.asarray(poses, dtype=np.float64)
    poses = poses / np.linalg.norm(poses, axis=1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(poses[:, 0]), 0, 1)))


class OrientationMetricsTests(unittest.TestCase):
    def run_cli(self, *arguments, succeeds=True):
        command = [PYTHON, '-m', *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        if succeeds:
            self.assertEqual(result.returncode, 0, 'Actual evaluation CLI must succeed:\n'
                             + ' '.join(command) + '\n' + result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, 'Invalid evaluation must fail')
        return result

    def export(self, checkpoint, directory):
        self.run_cli('model.broad_export_integer', '--checkpoint', str(checkpoint), '--output-dir', str(directory))

    def evaluate(self, dataset, source, output, mode, split='validation', succeeds=True):
        option = '--checkpoint' if mode == 'float' else '--export-dir'
        return self.run_cli('model.orientation_evaluate', '--dataset-dir', str(dataset), '--split', split,
                            option, str(source), '--output', str(output), succeeds=succeeds)

    def assert_metrics(self, actual, errors):
        errors = np.asarray(errors, dtype=np.float64)
        self.assertEqual(actual['sample_count'], len(errors))
        self.assertEqual(actual['above_5_deg_count'], int(np.count_nonzero(errors > 5)))
        expected = ({'full_rmse_deg': float(np.sqrt(np.mean(errors ** 2))),
                     'p50_deg': float(np.percentile(errors, 50)), 'p95_deg': float(np.percentile(errors, 95)),
                     'max_deg': float(errors.max())} if len(errors) else
                    dict.fromkeys(('full_rmse_deg', 'p50_deg', 'p95_deg', 'max_deg')))
        for name, value in expected.items():
            if value is None:
                self.assertIsNone(actual[name])
            else:
                self.assertAlmostEqual(actual[name], value, delta=1e-5)

    def test_float_and_integer_metrics_are_sample_weighted_sign_invariant_and_keep_cold_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequences = [('hold', poses_z([0, 0, 0, 0])), ('single_axis', -poses_z([90, 90, 90])),
                         ('hold', poses_z([180, 180]))]
            sequences[0][1][1::2] *= -1
            dataset = root / 'dataset'
            manifest = write_dataset(dataset, sequences)
            checkpoint, exported = root / 'checkpoint.pt', root / 'export'
            write_checkpoint(checkpoint, dataset)
            self.export(checkpoint, exported)
            for mode, source in (('float', checkpoint), ('integer', exported)):
                with self.subTest(mode=mode):
                    output = root / mode / 'report.json'
                    self.evaluate(dataset, source, output, mode)
                    report = json.loads(output.read_text())
                    self.assertIn('total_sample_count', report, 'Report must separate all frames from scored frames')
                    self.assertEqual((report['mode'], report['total_sample_count'], report['warmup_samples']),
                                     (mode, 9, 1))
                    self.assert_metrics(report, [0, 0, 0, 90, 90, 180])
                    self.assert_metrics(report['cold_start'], [0, 90, 180])
                    self.assert_metrics(report['stationary'], [0, 0, 0, 90, 90, 180])
                    self.assert_metrics(report['moving'], [])
                    self.assert_metrics(report['families']['hold'], [0, 0, 0, 180])
                    self.assert_metrics(report['families']['single_axis'], [90, 90])
                    self.assertEqual(len(report['sequences']), 3)
                    for result, entry, (_, poses) in zip(report['sequences'], manifest['sequences'], sequences):
                        self.assertEqual(result['sequence_id'], entry['sequence_id'])
                        errors = errors_for_identity(poses)
                        self.assert_metrics(result, errors[1:])
                        with np.load(output.parent / result['predictions_file'], allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved['targets'], poses)
                            np.testing.assert_allclose(saved['angle_errors_deg'], errors, atol=1e-5)
                            np.testing.assert_array_equal(saved['scored_mask'], [False] + [True] * (len(poses)-1))
                            np.testing.assert_array_equal(saved['moving_mask'], np.zeros(len(poses), dtype=bool))
                            np.testing.assert_allclose(saved['predictions'], np.tile([1, 0, 0, 0], (len(poses), 1)))
                            self.assertEqual(saved['hidden'].shape, (len(poses), 96))
                            self.assertEqual(saved['raw_outputs'].shape, (len(poses), 4))
                            if mode == 'integer':
                                for name in ('quantized_inputs', 'hidden', 'raw_outputs'):
                                    self.assertEqual(saved[name].dtype, np.dtype('int16'))

    def test_motion_uses_pose_changes_clipping_counts_all_frames_and_invalid_scores_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            poses = poses_z([0, 0, .01, .01, .02])
            dataset = root / 'dataset'
            manifest = write_dataset(dataset, [('stop_go', poses)])
            data_path = dataset / manifest['sequences'][0]['path']
            with np.load(data_path, allow_pickle=False) as saved:
                sensors = saved['sensors'].copy()
            sensors[1, 3] = .25  # Stationary gyro noise must not change the scoring class.
            sensors[2, 3:6] = 0  # Pose still moves when the gyro measurement misses it.
            np.savez_compressed(data_path, sensors=sensors, quaternion=poses)
            manifest['sequences'][0]['sha256'] = sha256(data_path)
            (dataset / 'manifest.json').write_text(json.dumps(manifest))
            checkpoint, exported = root / 'checkpoint.pt', root / 'export'
            write_checkpoint(checkpoint, dataset, std=.0001)
            self.export(checkpoint, exported)
            output = root / 'motion/report.json'
            self.evaluate(dataset, exported, output, 'integer')
            report = json.loads(output.read_text())
            self.assertIn('moving', report, 'Report must classify movement from target quaternion deltas')
            errors = errors_for_identity(poses)
            self.assert_metrics(report['moving'], errors[[2, 4]])
            self.assert_metrics(report['stationary'], errors[[1, 3]])
            entry = report['sequences'][0]
            with np.load(dataset / 'validation-0000.npz', allow_pickle=False) as data:
                unbounded = np.array([[half_away(value / .0001 * 2048) for value in row]
                                      for row in data['sensors']])
            quantized = np.array([[saturate(int(value)) for value in row] for row in unbounded], dtype=np.int16)
            self.assertEqual(report['input_clipping_count'], int(np.count_nonzero(unbounded != quantized)))
            with np.load(output.parent / entry['predictions_file'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['moving_mask'], [False, False, True, False, True])
                np.testing.assert_array_equal(saved['quantized_inputs'], quantized)
            zero_checkpoint, zero_export = root / 'zero.pt', root / 'zero-export'
            write_checkpoint(zero_checkpoint, dataset, zero=True)
            self.export(zero_checkpoint, zero_export)
            empty_dataset = root / 'all-warmup'
            write_dataset(empty_dataset, [('hold', poses_z([0]))])
            for mode, zero_source, good_source in (('float', zero_checkpoint, checkpoint),
                                                  ('integer', zero_export, exported)):
                for reason, data, source in (('zero', dataset, zero_source),
                                              ('empty', empty_dataset, good_source)):
                    with self.subTest(mode=mode, reason=reason):
                        invalid_output = root / f'{mode}-{reason}/report.json'
                        result = self.evaluate(data, source, invalid_output, mode, succeeds=False)
                        self.assertRegex(result.stderr.lower(), 'zero|scor|warmup')
                        self.assertFalse(invalid_output.exists(), 'Invalid predictions must not get a success report')

    def test_confirmation_binds_evaluation_data_separately_and_rejects_stale_float_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, confirmation = root / 'training', root / 'confirmation'
            write_dataset(training, [('hold', poses_z([0, 0]))], split='train')
            write_dataset(confirmation, [('hold', poses_z([90, 90]))], split='confirmation')
            checkpoint, exported = root / 'checkpoint.pt', root / 'export'
            write_checkpoint(checkpoint, training)
            self.export(checkpoint, exported)
            for mode, source in (('float', checkpoint), ('integer', exported)):
                with self.subTest(mode=mode):
                    output = root / mode / 'report.json'
                    self.evaluate(confirmation, source, output, mode, split='confirmation')
                    report = json.loads(output.read_text())
                    self.assertEqual(report['split'], 'confirmation')
                    self.assertEqual(report['dataset_manifest_sha256'], sha256(confirmation / 'manifest.json'))
                    self.assertEqual(report['training_dataset_manifest_sha256'], sha256(training / 'manifest.json'))
                    self.assertNotEqual(report['dataset_manifest_sha256'], report['training_dataset_manifest_sha256'])
                    self.assertEqual(report['checkpoint_sha256'], sha256(checkpoint))
                    self.assertEqual(report['sources']['model/orientation_evaluate.py'],
                                     sha256(ROOT / 'model/orientation_evaluate.py'))
                    if mode == 'integer':
                        self.assertEqual(report['export_manifest_sha256'], sha256(exported / 'manifest.json'))
                    else:
                        stale = torch.load(checkpoint, map_location='cpu', weights_only=True)
                        stale['training_recipe_sources']['model/orientation_train.py'] = '0' * 64
                        stale_path = root / 'stale.pt'
                        torch.save(stale, stale_path)
                        stale_output = root / 'stale/report.json'
                        result = self.evaluate(confirmation, stale_path, stale_output, 'float',
                                               split='confirmation', succeeds=False)
                        self.assertRegex(result.stderr.lower(), 'recipe|source|sha256')
                        self.assertFalse(stale_output.exists())


if __name__ == '__main__':
    unittest.main()
