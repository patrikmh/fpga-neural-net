"""THICKENING: causal, physical motion-prediction datasets.

Behaviors covered:
1. Future pose labels use exactly 50 ms, including fractional steps and quaternion signs.
2. Default families supply bounded acceleration, changing axes and actual reversals.
3. Seeded whole sequences retain their prefixes when lengths or split counts change.
4. The loader rejects incompatible semantics and broken provenance before training.
5. Invalid horizons and trajectories without future truth are rejected explicitly.
"""

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from test_orientation_sim import multiply, reference_sensors


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
DT = .0035
FAMILIES = ['hold', 'spin', 'smooth_acceleration', 'changing_axis', 'reversal']


def read_dataset(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    sequences = {}
    for entry in manifest['sequences']:
        with np.load(directory / entry['path'], allow_pickle=False) as saved:
            sequences[entry['sequence_id']] = (entry, {name: saved[name].copy() for name in saved.files})
    return manifest, sequences


class MotionDatasetTests(unittest.TestCase):
    def generate(self, directory, *arguments):
        command = [PYTHON, '-m', 'model.motion_generate', '--output-dir', str(directory),
                   '--seed', '17', '--horizon-ms', '50', *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, 'Motion dataset CLI must succeed:\n' + result.stdout + result.stderr)
        return read_dataset(directory)

    def test_spin_labels_are_exactly_fifty_milliseconds_ahead_with_fractional_steps_and_sign_invariance(self):
        from model.motion_generate import future_targets

        with tempfile.TemporaryDirectory() as temporary:
            _, sequences = self.generate(Path(temporary) / 'spin', '--train-sequences', '1',
                                          '--validation-sequences', '0', '--families', 'spin', '--length', '96')
            _, arrays = next(iter(sequences.values()))
            current, targets, sensors = (arrays[name] for name in ('current_quaternion', 'quaternion', 'sensors'))
            self.assertEqual((current.shape, targets.shape, sensors.shape), ((81, 4), (81, 4), (81, 9)))
            omega = sensors[10, 3:6]
            rate = np.linalg.norm(omega)
            self.assertGreater(rate, .1, 'Analytic fixture must move')
            delta = np.r_[math.cos(rate * .05 / 2), omega / rate * math.sin(rate * .05 / 2)]
            expected = np.stack([multiply(pose, delta) for pose in current[10:]])
            alignment = np.sum(expected * targets[10:], axis=1)
            np.testing.assert_allclose(np.abs(alignment), 1, atol=1e-11, rtol=0)
            signed = current.copy()
            signed[::2] *= -1
            regular = future_targets(current, .05)
            flipped = future_targets(signed, .05)
            np.testing.assert_allclose(np.abs(np.sum(regular * flipped, axis=1)), 1, atol=1e-12)

    def test_default_families_have_physical_sensors_and_distinct_smooth_temporal_dynamics(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, sequences = self.generate(Path(temporary) / 'families', '--train-sequences', '5',
                                          '--validation-sequences', '0', '--length', '1024')
            self.assertEqual([entry['family'] for entry, _ in sequences.values()], FAMILIES)
            initial = []
            for entry, arrays in sequences.values():
                with self.subTest(family=entry['family']):
                    poses, sensors = arrays['current_quaternion'], arrays['sensors']
                    initial.append(poses[0])
                    np.testing.assert_allclose(np.linalg.norm(poses, axis=1), 1, atol=1e-10)
                    np.testing.assert_allclose(sensors[:4, 3:6], 0, atol=1e-9)
                    expected = np.stack([reference_sensors(poses[max(0, i-1)], pose)
                                         for i, pose in enumerate(poses)])
                    np.testing.assert_allclose(sensors, expected, atol=1e-8, rtol=1e-9)
                    omega = sensors[4:, 3:6]
                    speed = np.linalg.norm(omega, axis=1)
                    self.assertLessEqual(float(speed.max()), 6 + 1e-8)
                    family = entry['family']
                    if family == 'hold':
                        np.testing.assert_allclose(speed, 0, atol=1e-9)
                    elif family == 'spin':
                        self.assertGreater(float(speed.min()), .1)
                        self.assertLess(float(speed.max() - speed.min()), 1e-7)
                        self.assertAlmostEqual(float(speed.max()), 6, delta=1e-8,
                                               msg='Spin control must cover the app slew limit of 6 rad/s')
                    else:
                        acceleration = np.linalg.norm(np.diff(omega, axis=0), axis=1) / DT
                        self.assertGreater(float(acceleration.max()), .1)
                        self.assertLess(float(acceleration.max()), 100,
                                        'Smooth families must not jump angular velocity instantaneously')
                        singular = np.linalg.svd(omega, compute_uv=False)
                        if family == 'changing_axis':
                            self.assertGreater(float(singular[1]), .05 * float(singular[0]))
                        else:
                            self.assertLess(float(singular[1]), 1e-6 * float(singular[0]))
                        if family == 'reversal':
                            axis = omega[np.argmax(speed)] / speed.max()
                            signed_rate = omega @ axis
                            self.assertLess(float(signed_rate.min()), -.2)
                            self.assertGreater(float(signed_rate.max()), .2)
                        if family == 'smooth_acceleration':
                            self.assertGreater(float(speed.max() - speed.min()), .5)
            self.assertTrue(all(abs(float(pose[0])) < .999 for pose in initial))
            self.assertGreater(float(np.linalg.norm(initial[0] - initial[1])), .01)

    def test_seeded_sequences_are_independent_of_length_other_counts_and_unopened_splits(self):
        from model.motion_generate import load_sequences

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, baseline = self.generate(root / 'short', '--train-sequences', '5',
                                         '--validation-sequences', '1', '--length', '96')
            _, extended = self.generate(root / 'long', '--train-sequences', '7',
                                         '--validation-sequences', '3', '--confirmation-sequences', '2',
                                         '--length', '160')
            _, repeated = self.generate(root / 'repeat', '--train-sequences', '5',
                                         '--validation-sequences', '1', '--length', '96')
            for sequence_id, (entry, arrays) in baseline.items():
                with self.subTest(sequence_id=sequence_id):
                    self.assertTrue(sequence_id.startswith(entry['split'] + '-'))
                    for name, values in arrays.items():
                        np.testing.assert_array_equal(repeated[sequence_id][1][name], values)
                        np.testing.assert_array_equal(extended[sequence_id][1][name][:len(values)], values)
            manifest, _ = read_dataset(root / 'long')
            self.assertEqual(len({entry['sequence_id'] for entry in manifest['sequences']}), 12)
            for entry in manifest['sequences']:
                if entry['split'] != 'train':
                    (root / 'long' / entry['path']).unlink()
            self.assertEqual(len(load_sequences(root / 'long', 'train')), 7)

    def test_loader_rejects_incompatible_units_timing_convention_paths_and_identity(self):
        from model.motion_generate import load_sequences

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pristine = root / 'pristine'
            self.generate(pristine, '--train-sequences', '1', '--validation-sequences', '0', '--length', '64')
            mutations = ('dt', 'units', 'convention', 'path', 'hash', 'source')
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    directory = root / mutation
                    shutil.copytree(pristine, directory)
                    manifest_path = directory / 'manifest.json'
                    manifest = json.loads(manifest_path.read_text())
                    if mutation == 'dt':
                        manifest['dt_s'] = .007
                    elif mutation == 'units':
                        manifest['sensor_units'] = {'acc': 'g', 'gyr': 'deg/s', 'mag': 'uT'}
                    elif mutation == 'convention':
                        manifest['quaternion_convention'] = 'xyzw_ENU_to_body'
                    elif mutation == 'path':
                        outside = root / 'outside.npz'
                        shutil.copyfile(directory / manifest['sequences'][0]['path'], outside)
                        manifest['sequences'][0]['path'] = '../outside.npz'
                    elif mutation == 'hash':
                        manifest['sequences'][0]['sha256'] = '0' * 64
                    else:
                        manifest['sources']['model/motion_generate.py'] = '0' * 64
                    manifest_path.write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError, msg=f'{mutation} must be rejected before model fitting'):
                        load_sequences(directory, 'train')

    def test_invalid_horizons_and_trajectories_without_future_truth_are_rejected(self):
        from model.motion_generate import future_targets

        poses = np.tile([1., 0., 0., 0.], (16, 1))
        for horizon in (0., -.05, float('nan'), float('inf'), float('-inf')):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    future_targets(poses, horizon)
        with self.assertRaises(ValueError):
            future_targets(poses[:15], .05)
        with tempfile.TemporaryDirectory() as temporary:
            for index, horizon in enumerate(('0', '-50', 'nan', 'inf')):
                with self.subTest(cli_horizon=horizon):
                    result = subprocess.run([PYTHON, '-m', 'model.motion_generate', '--output-dir',
                                             str(Path(temporary) / str(index)), '--train-sequences', '1',
                                             '--validation-sequences', '0', '--length', '64',
                                             '--horizon-ms', horizon], cwd=ROOT, capture_output=True, text=True,
                                            timeout=30)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('finite and positive', result.stderr)
