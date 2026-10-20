"""THICKENING: physical and reproducible synthetic orientation datasets.

Behaviors covered:
1. The CLI produces five declared motion families with independently correct sensors.
2. Whole-sequence splits reproduce independently and support unopened confirmation data.
3. The loader validates selected data identities/structure without opening other splits.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from test_orientation_sim import reference_sensors


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
FAMILIES = ['hold', 'single_axis', 'mixed', 'stop_go', 'spin']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sequences(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    sequences = {}
    for entry in manifest['sequences']:
        with np.load(directory / entry['path'], allow_pickle=False) as saved:
            sequences[entry['sequence_id']] = (entry, saved['sensors'].copy(), saved['quaternion'].copy())
    return manifest, sequences


class OrientationDatasetTests(unittest.TestCase):
    def generate(self, directory, *arguments):
        command = [PYTHON, '-m', 'model.orientation_generate', '--output-dir', str(directory),
                   '--seed', '17', *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, 'Dataset CLI must succeed:\n' + ' '.join(command)
                         + '\n' + result.stdout + result.stderr)
        return read_sequences(directory)

    def test_motion_families_have_correct_body_sensors_bounded_continuous_poses_and_holds(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'dataset'
            manifest, sequences = self.generate(directory, '--train-sequences', '5',
                                                '--validation-sequences', '1', '--length', '2048')
            self.assertIn('config', manifest, 'Manifest must declare the motion-family generation recipe')
            self.assertEqual(manifest['config']['families'], FAMILIES)
            self.assertEqual(manifest['config']['train_sequences'], 5)
            self.assertEqual(manifest['config']['validation_sequences'], 1)
            self.assertEqual(manifest['config']['confirmation_sequences'], 0)
            self.assertEqual(manifest['dt_s'], .0035)
            self.assertEqual(manifest['max_angular_rate_rad_s'], 6)
            self.assertEqual(manifest['sensor_order'],
                             ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z'])
            self.assertEqual(manifest['sensor_units'], {'acc': 'm/s^2', 'gyr': 'rad/s', 'mag': 'uT'})
            self.assertEqual(manifest['quaternion_convention'], 'wxyz_body_to_ENU')
            for name in ('model/orientation_generate.py', 'host/orientation_sim.py'):
                self.assertEqual(manifest['sources'][name], sha256(ROOT / name))
            training = [item for item in sequences.values() if item[0]['split'] == 'train']
            self.assertEqual({item[0]['family'] for item in training}, set(FAMILIES))
            initial = []
            for entry, sensors, poses in training:
                with self.subTest(family=entry['family']):
                    self.assertEqual(entry['sha256'], sha256(directory / entry['path']))
                    self.assertEqual(sensors.shape, (2048, 9))
                    self.assertEqual(poses.shape, (2048, 4))
                    np.testing.assert_allclose(np.linalg.norm(poses, axis=1), 1, atol=1e-10)
                    self.assertTrue(np.isfinite(sensors).all())
                    np.testing.assert_allclose(sensors[0, 3:6], 0, atol=1e-10)
                    dots = np.sum(poses[:-1] * poses[1:], axis=1)
                    self.assertTrue(np.all(dots >= 0), 'Quaternion signs must stay continuous across full turns')
                    rate = np.linalg.norm(sensors[:, 3:6], axis=1)
                    self.assertLessEqual(float(rate.max()), 6 + 1e-8)
                    expected = np.stack([reference_sensors(poses[max(0, index-1)], pose)
                                         for index, pose in enumerate(poses)])
                    np.testing.assert_allclose(sensors, expected, atol=1e-8, rtol=1e-9)
                    initial.append(poses[0])
                    moving = rate > 1e-7
                    family = entry['family']
                    if family == 'hold':
                        self.assertFalse(moving.any())
                        np.testing.assert_allclose(poses, np.tile(poses[0], (len(poses), 1)), atol=1e-10)
                    elif family == 'single_axis':
                        self.assertTrue(moving.any())
                        singular = np.linalg.svd(sensors[moving, 3:6], compute_uv=False)
                        self.assertLess(float(singular[1]), float(singular[0]) * 1e-7)
                        self.assertFalse(moving[-100:].any(), 'A single-axis maneuver ends with a long hold')
                    elif family == 'mixed':
                        singular = np.linalg.svd(sensors[moving, 3:6], compute_uv=False)
                        self.assertGreater(float(singular[1]), float(singular[0]) * .01)
                    elif family == 'stop_go':
                        starts = np.concatenate(([True], moving[1:] != moving[:-1]))
                        runs = moving[starts]
                        self.assertGreaterEqual(int(runs.sum()), 2)
                        self.assertGreaterEqual(int((~runs).sum()), 2)
                    elif family == 'spin':
                        self.assertAlmostEqual(float(rate.max()), 6, delta=1e-8,
                                               msg='Spin must cover the demo target-slew rate of 6 rad/s')
                        self.assertGreaterEqual(float(rate.sum() * .0035), 2 * np.pi)
            self.assertTrue(all(abs(float(pose[0])) < .999 for pose in initial))
            self.assertGreater(float(np.linalg.norm(initial[0] - initial[1])), .01)

    def test_seeded_arrays_and_split_ids_are_stable_when_other_split_counts_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = ['--train-sequences', '2', '--validation-sequences', '1', '--length', '40']
            manifest, baseline = self.generate(root / 'baseline', *arguments)
            _, repeated = self.generate(root / 'repeated', *arguments)
            _, more_validation = self.generate(root / 'more-validation', '--train-sequences', '2',
                                               '--validation-sequences', '3', '--confirmation-sequences', '2',
                                               '--length', '40')
            _, more_training = self.generate(root / 'more-training', '--train-sequences', '4',
                                             '--validation-sequences', '1', '--length', '40')
            self.assertEqual(set(baseline), set(repeated))
            for sequence_id, (entry, sensors, poses) in baseline.items():
                with self.subTest(sequence_id=sequence_id):
                    self.assertTrue(sequence_id.startswith(entry['split'] + '-'))
                    for other in (repeated, more_validation, more_training):
                        np.testing.assert_array_equal(other[sequence_id][1], sensors)
                        np.testing.assert_array_equal(other[sequence_id][2], poses)
            self.assertFalse(any(entry['split'] == 'confirmation' for entry in manifest['sequences']))
            self.assertEqual(len(more_validation), len(set(more_validation)))
            confirmation_manifest, confirmation = self.generate(
                root / 'confirmation-only', '--train-sequences', '0', '--validation-sequences', '0',
                '--confirmation-sequences', '1', '--families', 'spin', '--length', '1')
            self.assertEqual(len(confirmation_manifest['sequences']), 1)
            entry, sensors, poses = next(iter(confirmation.values()))
            self.assertEqual((entry['split'], entry['family']), ('confirmation', 'spin'))
            self.assertEqual((sensors.shape, poses.shape), ((1, 9), (1, 4)))
            np.testing.assert_allclose(sensors[0, 3:6], 0, atol=1e-10)

    def test_selected_loader_rejects_tampering_and_invalid_arrays_without_opening_other_splits(self):
        from model.orientation_generate import load_sequences

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pristine = root / 'pristine'
            self.generate(pristine, '--train-sequences', '1', '--validation-sequences', '1', '--length', '40')
            unread = root / 'unread-other-splits'
            shutil.copytree(pristine, unread)
            unread_manifest = json.loads((unread / 'manifest.json').read_text())
            for entry in unread_manifest['sequences']:
                if entry['split'] != 'train':
                    (unread / entry['path']).unlink()
            unread_manifest['sequences'].append({'split': 'confirmation', 'sequence_id': 'confirmation-0000',
                                                 'family': 'hold', 'path': 'unopened-confirmation.npz',
                                                 'sha256': '0' * 64})
            (unread / 'manifest.json').write_text(json.dumps(unread_manifest))
            self.assertEqual(len(load_sequences(unread, 'train')), 1)

            mutations = ('hash', 'source', 'duplicate_id', 'escape', 'symlink',
                         'sensors_shape', 'quaternion_shape', 'nonfinite_sensors',
                         'nonfinite_quaternion', 'nonunit_quaternion')
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    directory = root / mutation
                    shutil.copytree(pristine, directory)
                    manifest_path = directory / 'manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    entry = next(item for item in manifest['sequences'] if item['split'] == 'train')
                    path = directory / entry['path']
                    if mutation == 'hash':
                        entry['sha256'] = '0' * 64
                    elif mutation == 'source':
                        manifest.setdefault('sources', {})['model/orientation_generate.py'] = '0' * 64
                    elif mutation == 'duplicate_id':
                        manifest['sequences'].append(dict(entry, split='confirmation'))
                    elif mutation in ('escape', 'symlink'):
                        external = root / (mutation + '-outside.npz')
                        shutil.copyfile(path, external)
                        if mutation == 'escape':
                            entry['path'] = '../' + external.name
                        else:
                            link = directory / 'outside-link.npz'
                            link.symlink_to(external)
                            entry['path'] = link.name
                    else:
                        with np.load(path, allow_pickle=False) as saved:
                            sensors, poses = saved['sensors'].copy(), saved['quaternion'].copy()
                        if mutation == 'sensors_shape':
                            sensors = sensors[:, :8]
                        elif mutation == 'quaternion_shape':
                            poses = poses[:-1]
                        elif mutation == 'nonfinite_sensors':
                            sensors[0, 0] = np.inf
                        elif mutation == 'nonfinite_quaternion':
                            poses[0, 0] = np.nan
                        elif mutation == 'nonunit_quaternion':
                            poses[0] *= 2
                        np.savez_compressed(path, sensors=sensors, quaternion=poses)
                        entry['sha256'] = sha256(path)
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError, msg='Reject invalid dataset: ' + mutation):
                        load_sequences(directory, 'train')


if __name__ == '__main__':
    unittest.main()
