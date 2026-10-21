"""THICKENING: a declared reset pose is a model domain, not hidden truth.

Behaviors covered:
1. Identity-reset and random-reset datasets share dynamics while retaining continuous poses.
2. Training binds the reset domain and evaluation separates qualification from explicit stress tests.
"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
TASK = {'kind': 'motion_prediction', 'horizon_s': .05, 'model_dt_s': .0035,
        'target': 'absolute_body_to_enu_wxyz'}


class MotionInitialPoseTests(unittest.TestCase):
    def command(self, module, *arguments, success=True):
        result = subprocess.run([PYTHON, '-m', module, *map(str, arguments)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, f'{module} must succeed:\n{result.stdout}{result.stderr}')
        return result

    def generate(self, directory, *arguments):
        self.command('model.motion_generate', '--output-dir', directory, '--seed', '17',
                     '--train-sequences', '1', '--validation-sequences', '1',
                     '--length', '48', '--horizon-ms', '50', *arguments)
        return json.loads((directory / 'manifest.json').read_text())

    def test_reset_pose_changes_only_pose_domain_and_preserves_full_turn_quaternion_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = ['--train-sequences', '1', '--validation-sequences', '0',
                      '--families', 'spin', '--length', '1200']
            default = self.generate(root / 'default', *shared)
            self.assertEqual(default.get('initial_pose'), 'random', 'Existing default domain must stay random')
            explicit = self.generate(root / 'random', *shared, '--initial-pose', 'random')
            identity = self.generate(root / 'identity', *shared, '--initial-pose', 'identity')
            self.assertEqual(identity['initial_pose'], 'identity')
            arrays = []
            for directory, manifest in ((root / 'default', default), (root / 'random', explicit),
                                        (root / 'identity', identity)):
                with np.load(directory / manifest['sequences'][0]['path']) as saved:
                    arrays.append({name: saved[name].copy() for name in saved.files})
            random_default, random_explicit, known = arrays
            for name, values in random_default.items():
                np.testing.assert_array_equal(random_explicit[name], values)
            np.testing.assert_allclose(known['current_quaternion'][:4], np.tile([1., 0., 0., 0.], (4, 1)), atol=1e-12)
            self.assertLess(abs(float(random_default['current_quaternion'][0, 0])), .999)
            np.testing.assert_allclose(known['sensors'][:, 3:6], random_default['sensors'][:, 3:6], atol=1e-9, rtol=1e-9,
                                       err_msg='Changing reset pose must not change sampled angular dynamics')
            poses = known['current_quaternion']
            np.testing.assert_allclose(np.linalg.norm(poses, axis=1), 1, atol=1e-10)
            self.assertTrue(np.all(np.sum(poses[1:] * poses[:-1], axis=1) > 0),
                            'Quaternion signs must remain continuous through full turns')
            self.assertLess(float(poses[:, 0].min()), -.9,
                            'Identity-start spin must cross the negative hemisphere without sign canonicalization')
            self.assertGreater(float(np.linalg.norm(known['sensors'][:, 3:6], axis=1).sum() * .0035), 2 * np.pi)

    def test_training_and_evaluation_bind_initial_pose_and_explicitly_disqualify_domain_stress_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, training = root / 'identity-train', root / 'training'
            self.generate(data, '--initial-pose', 'identity')
            training_args = ['--dataset-dir', data, '--epochs', '1', '--hidden-size', '96',
                             '--warmup', '4', '--chunk-size', '32', '--batch-size', '2', '--seed', '23',
                             '--learning-rate', '.001']
            self.command('model.motion_train', *training_args, '--output-dir', training)
            checkpoint = torch.load(training / 'checkpoint.pt', weights_only=True)
            report = json.loads((training / 'report.json').read_text())
            self.assertEqual(checkpoint['config']['initial_pose'], 'identity')
            self.assertEqual(report['config']['initial_pose'], 'identity')
            self.assertEqual(checkpoint['config']['task'], TASK)
            identity_eval, random_eval = root / 'identity-eval', root / 'random-eval'
            self.generate(identity_eval, '--initial-pose', 'identity', '--seed', '24')
            self.generate(random_eval, '--initial-pose', 'random', '--seed', '24')
            same_output = root / 'same-domain' / 'report.json'
            self.command('model.motion_evaluate', '--dataset-dir', identity_eval,
                         '--checkpoint', training / 'checkpoint.pt', '--output', same_output)
            same = json.loads(same_output.read_text())
            self.assertEqual((same['initial_pose'], same['training_initial_pose'], same['domain_match'],
                              same['qualification_eligible']), ('identity', 'identity', True, True))
            rejected_output = root / 'rejected-domain' / 'report.json'
            rejected = self.command('model.motion_evaluate', '--dataset-dir', random_eval,
                                     '--checkpoint', training / 'checkpoint.pt', '--output', rejected_output,
                                     success=False)
            self.assertNotEqual(rejected.returncode, 0, 'Unrequested reset-domain mismatch must fail')
            self.assertRegex(rejected.stderr.lower(), r'initial.pose|domain')
            self.assertFalse(rejected_output.exists())
            stress_output = root / 'stress-domain' / 'report.json'
            self.command('model.motion_evaluate', '--dataset-dir', random_eval,
                         '--checkpoint', training / 'checkpoint.pt', '--output', stress_output,
                         '--allow-initial-pose-mismatch')
            stress = json.loads(stress_output.read_text())
            self.assertEqual((stress['initial_pose'], stress['training_initial_pose'], stress['domain_match'],
                              stress['qualification_eligible']), ('random', 'identity', False, False))
            self.assertEqual(stress['task'], TASK)
            incompatible = {**checkpoint, 'config': {**checkpoint['config'], 'initial_pose': 'random'}}
            incompatible_path = root / 'incompatible-initial.pt'
            torch.save(incompatible, incompatible_path)
            initial = self.command('model.motion_train', *training_args, '--initial-checkpoint', incompatible_path,
                                    '--output-dir', root / 'rejected-initial', success=False)
            self.assertNotEqual(initial.returncode, 0, 'Warm-starting a different reset domain must fail')
            self.assertRegex(initial.stderr.lower(), r'initial.pose|domain')
