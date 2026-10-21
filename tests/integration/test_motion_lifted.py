"""THICKENING: signed continuous-quaternion learning without changing frozen recipes.

Behaviors covered:
1. Signed raw loss learns the continuous target lift, including from zero/opposite output.
2. Chronological state crosses full turns consistently and invalid lift domains are rejected.
3. The separate trainer exports/evaluates the unchanged H96 model with complete provenance.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from model.broad_integer import sha256
from model.broad_network import BroadPredictor


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')


class MotionLiftedTests(unittest.TestCase):
    def trainer(self):
        self.assertIsNotNone(importlib.util.find_spec('model.motion_train_lifted'),
                             'Continuous signed prediction needs the separate motion_train_lifted trainer')
        return importlib.import_module('model.motion_train_lifted')

    def command(self, module, *arguments, success=True):
        result = subprocess.run([PYTHON, '-m', module, *map(str, arguments)], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, f'{module} must succeed:\n{result.stdout}{result.stderr}')
        return result

    def generate(self, directory, *arguments):
        self.command('model.motion_generate', '--output-dir', directory, '--seed', '17', '--initial-pose', 'identity',
                     '--train-sequences', '3', '--validation-sequences', '2', '--length', '64', *arguments)

    def train(self, directory, output, *arguments, success=True):
        return self.command('model.motion_train_lifted', '--dataset-dir', directory, '--output-dir', output,
                            '--epochs', '1', '--hidden-size', '96', '--warmup', '4', '--chunk-size', '16',
                            '--batch-size', '2', '--seed', '53', '--learning-rate', '.001', *arguments,
                            success=success)

    def test_signed_loss_has_zero_and_antipodal_gradients_toward_the_continuous_target(self):
        loss_fn = self.trainer().forecast_loss
        targets = torch.tensor([[[1., 0., 0., 0.]] * 3])
        mask = torch.tensor([[True, True, False]])
        raw = torch.tensor([[[0., 0., 0., 0.], [-1., 0., 0., 0.], [.3, .4, .5, .6]]], requires_grad=True)
        loss = loss_fn(raw, targets, mask)
        self.assertTrue(bool(torch.isfinite(loss)), 'Zero raw output must remain trainable')
        loss.backward()
        self.assertLess(float(raw.grad[0, 0, 0]), 0, 'Zero output must move toward the signed target')
        self.assertLess(float(raw.grad[0, 1, 0]), 0, 'Antipodal output must move through raw space toward the signed target')
        torch.testing.assert_close(raw.grad[0, 2], torch.zeros(4), rtol=0, atol=0)
        correct = loss_fn(targets, targets, mask)
        opposite = loss_fn(-targets, targets, mask)
        self.assertAlmostEqual(float(correct), 0, places=7)
        self.assertGreater(float(opposite), float(correct) + .1, 'Antipodal targets are physically equal but distinct continuous lift labels')
        with self.assertRaises(ValueError):
            loss_fn(torch.full_like(targets, float('nan')), targets, mask)

    def test_carried_sequence_survives_full_turns_and_trainer_rejects_invalid_lift_domains(self):
        module = self.trainer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full_turn = root / 'full-turn'
            self.generate(full_turn, '--families', 'spin', '--train-sequences', '1',
                          '--validation-sequences', '1', '--length', '400')
            manifest = json.loads((full_turn / 'manifest.json').read_text())
            with np.load(full_turn / manifest['sequences'][0]['path']) as saved:
                poses, sensors = saved['quaternion'], saved['sensors']
            self.assertLess(float(poses[:, 0].min()), -.9, 'Fixture must cross the quaternion hemisphere')
            self.assertTrue(np.all(np.sum(poses[1:] * poses[:-1], axis=1) > 0))
            inputs = torch.tensor(sensors / np.array([9.81] * 3 + [6.] * 3 + [45.] * 3), dtype=torch.float32)[None]
            torch.manual_seed(53)
            network = BroadPredictor(hidden_size=96)
            with torch.no_grad():
                whole, final = module.raw_sequence(network, inputs)
                state, chunks = None, []
                for start in range(0, inputs.shape[1], 37):
                    values, state = module.raw_sequence(network, inputs[:, start:start+37], state)
                    chunks.append(values)
            torch.testing.assert_close(torch.cat(chunks, dim=1), whole, rtol=0, atol=0)
            torch.testing.assert_close(state, final, rtol=0, atol=0)
            random_domain = root / 'random'
            self.generate(random_domain, '--initial-pose', 'random')
            rejected = self.train(random_domain, root / 'rejected-random', success=False)
            self.assertNotEqual(rejected.returncode, 0, 'Signed identity lift cannot silently accept random reset poses')
            self.assertRegex(rejected.stderr.lower(), r'identity|initial.pose|domain')
            flipped = root / 'flipped'
            shutil.copytree(full_turn, flipped)
            manifest = json.loads((flipped / 'manifest.json').read_text())
            entry = manifest['sequences'][0]
            path = flipped / entry['path']
            with np.load(path) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays['quaternion'][20] *= -1
            np.savez_compressed(path, **arrays)
            entry['sha256'] = sha256(path)
            (flipped / 'manifest.json').write_text(json.dumps(manifest))
            rejected = self.train(flipped, root / 'rejected-flip', success=False)
            self.assertNotEqual(rejected.returncode, 0, 'A discontinuous target sign cannot masquerade as a continuous lift')
            self.assertRegex(rejected.stderr.lower(), r'continu|sign|lift')

    def test_separate_signed_trainer_preserves_h96_export_and_source_bound_evaluation(self):
        self.trainer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, training, export = (root / name for name in ('data', 'training', 'export'))
            self.generate(data)
            self.train(data, training)
            checkpoint = torch.load(training / 'checkpoint.pt', weights_only=True)
            config = checkpoint['config']
            self.assertEqual([config[key] for key in ('input_size', 'hidden_size', 'output_size')], [9, 96, 4])
            self.assertEqual(config['initial_pose'], 'identity')
            self.assertEqual(config['loss'], 'signed_raw_quaternion_mse')
            self.assertEqual(config['target_sign'], 'continuous_identity_anchored')
            self.assertEqual(config['task'], {'kind': 'motion_prediction', 'horizon_s': .05,
                                             'model_dt_s': .0035, 'target': 'absolute_body_to_enu_wxyz'})
            for name in ('model/motion_train_lifted.py', 'model/motion_train.py', 'model/motion_generate.py',
                         'model/motion_evaluate.py', 'model/orientation_generate.py',
                         'model/orientation_evaluate.py', 'host/orientation_sim.py'):
                self.assertEqual(checkpoint['training_recipe_sources'][name], sha256(ROOT / name))
            report = json.loads((training / 'report.json').read_text())
            self.assertEqual(report['training_samples_scored'], 3 * (49 - 4))
            self.assertEqual(report['optimizer_steps'], 8)
            self.assertEqual(report['selected_epoch'], min(report['history'], key=lambda row: row['validation']['full_rmse_deg'])['epoch'])
            self.command('model.broad_export_integer', '--checkpoint', training / 'checkpoint.pt', '--output-dir', export)
            for mode, arguments in (('float', ['--checkpoint', training / 'checkpoint.pt']),
                                    ('integer', ['--export-dir', export])):
                output = root / mode / 'report.json'
                self.command('model.motion_evaluate', '--dataset-dir', data, '--split', 'validation',
                             *arguments, '--output', output)
                evaluated = json.loads(output.read_text())
                self.assertEqual(evaluated['sample_count'], 2 * (49 - 4))
                self.assertTrue(evaluated['qualification_eligible'])
                self.assertEqual(evaluated['task'], config['task'])
                self.assertEqual(evaluated['checkpoint_sha256'], sha256(training / 'checkpoint.pt'))
