"""THICKENING: verified forecasts mature only when chronological truth arrives.

Behaviors covered:
1. A pending chip forecast is scored against interpolated future truth and issue-time baselines.
2. Verified RESET discards outstanding forecasts before a fresh prediction sequence.
3. A bad device response cannot advance time or invent a matured forecast.
"""

import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request

import numpy as np

from model.broad_integer import GoldenBroadPredictor, quantize_inputs
from test_orientation_sim import multiply, reference_pose_step, reference_sensors, unit
from integration.test_orientation_demo import ROOT, crc_frame, rotated_native_app


def angle(a, b):
    return math.degrees(2 * math.acos(np.clip(abs(float(np.dot(unit(a), unit(b)))), 0, 1)))


def interpolate(a, b, fraction):
    a, b = unit(a), unit(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    half = math.acos(np.clip(dot, -1, 1))
    if half < 1e-10:
        return unit((1 - fraction) * a + fraction * b)
    return unit((math.sin((1 - fraction) * half) * a + math.sin(fraction * half) * b) / math.sin(half))


def issue_baselines(sensors):
    gravity, gyro, magnetic = np.asarray(sensors).reshape(3, 3)
    b1 = unit(gravity)
    b2 = unit(magnetic - (magnetic @ b1) * b1)
    earth_magnetic = reference_sensors([1, 0, 0, 0], [1, 0, 0, 0])[6:]
    w1 = np.array([0., 0., 1.])
    w2 = unit(earth_magnetic - (earth_magnetic @ w1) * w1)
    rotation = np.column_stack([w1, w2, np.cross(w1, w2)]) @ np.column_stack([b1, b2, np.cross(b1, b2)]).T
    w = math.sqrt(1 + np.trace(rotation)) / 2
    pose = unit([w, (rotation[2, 1] - rotation[1, 2]) / (4 * w),
                 (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                 (rotation[1, 0] - rotation[0, 1]) / (4 * w)])
    rate = np.linalg.norm(gyro)
    delta = np.array([1., 0., 0., 0.]) if rate < 1e-12 else np.r_[math.cos(rate * .05 / 2), gyro / rate * math.sin(rate * .05 / 2)]
    return {'sensor_hold': pose, 'sensor_constant_velocity': unit(multiply(pose, delta))}


class MotionDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='motion-demo-')
        cls.root = Path(cls.temporary.name)
        cls.export = cls.root / 'export'
        commands = [
            ('model.motion_generate', '--output-dir', cls.root / 'data', '--seed', '17', '--initial-pose', 'identity',
             '--train-sequences', '2', '--validation-sequences', '1', '--length', '48'),
            ('model.motion_train', '--dataset-dir', cls.root / 'data', '--output-dir', cls.root / 'training',
             '--epochs', '1', '--hidden-size', '96', '--warmup', '4', '--chunk-size', '32', '--batch-size', '2'),
            ('model.broad_export_integer', '--checkpoint', cls.root / 'training/checkpoint.pt', '--output-dir', cls.export),
        ]
        for module, *arguments in commands:
            result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', module, *map(str, arguments)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError(f'{module} fixture failed:\n{result.stdout}{result.stderr}')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def target(self, index):
        choices = ([.8, .4, .2, -.1], [.7, -.3, .5, .2], [.6, .2, -.3, .6])
        return unit(choices[(index // 4) % 3]).tolist()

    def stream(self, actions):
        predictor = GoldenBroadPredictor.from_export(self.export)
        mean, std = (np.asarray(predictor.manifest['normalization'][key]) for key in ('mean', 'std'))
        requests, replies, expected = [], [], []
        truth, index = unit([1, 0, 0, 0]), 0
        for target in actions:
            if target is None:
                predictor.reset()
                truth, index = unit([1, 0, 0, 0]), 0
                requests.append(crc_frame(struct.pack('<BBH9h', 0xa6, 2, 65535, *([0] * 9))))
                replies.append(crc_frame(struct.pack('<BBH96h4hI', 0xa6, 0, 65535, *([0] * 96), *([0] * 4), 0)))
                expected.append(None)
                continue
            previous = truth
            truth = reference_pose_step(previous, target)
            sensors = reference_sensors(previous, truth)
            inputs, _ = quantize_inputs((sensors - mean) / std)
            hidden, raw = predictor.step(inputs)
            requests.append(crc_frame(struct.pack('<BBH9h', 0xa6, 1, index, *inputs)))
            replies.append(crc_frame(struct.pack('<BBH96h4hI', 0xa6, 0, index, *hidden, *raw, 124046)))
            expected.append({'truth': truth.copy(), 'raw': raw, 'hidden': hidden, 'quaternion': unit(raw),
                             'baselines': issue_baselines(inputs.astype(np.float64) / 2048 * std + mean)})
            index += 1
        return b''.join(requests), replies, expected

    def post(self, base, path, payload=None):
        request = urllib.request.Request(base + path, data=json.dumps(payload or {}).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_forecast_matures_using_original_chip_output_and_fractional_future_truth(self):
        actions = [None] + [self.target(i) for i in range(20)]
        requests, replies, expected = self.stream(actions)
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory), requests, b''.join(replies), module='host.motion_demo',
                                    export_dir=self.export) as (base, trace):
                self.assertEqual(self.post(base, '/api/connect')[0], 200)
                first_pending = None
                for i, target in enumerate(actions[1:]):
                    code, state = self.post(base, '/api/step', {'target_wxyz': target})
                    self.assertEqual(code, 200, state)
                    sample = state['sample']
                    self.assertEqual(sample['hidden'], expected[i + 1]['hidden'])
                    self.assertEqual(sample['raw_outputs'], expected[i + 1]['raw'])
                    self.assertEqual(sample['forecast']['status'], 'pending')
                    self.assertIsNone(sample['forecast']['truth_wxyz'])
                    self.assertIsNone(sample['forecast']['error_deg'])
                    self.assertIn('scored_forecast', sample, 'Future truth needs a separate matured-forecast record')
                    self.assertEqual(state['pending_forecast_count'], min(i + 1, 15))
                    if i == 0:
                        first_pending = dict(sample['forecast'])
                    if i < 15:
                        self.assertIsNone(sample['scored_forecast'])
                        continue
                    issued = i - 15
                    scored = sample['scored_forecast']
                    original = expected[issued + 1]
                    truth = interpolate(expected[i]['truth'], expected[i + 1]['truth'], 2 / 7)
                    self.assertEqual(scored['issued_index'], issued)
                    self.assertAlmostEqual(scored['issued_at_s'], (issued + 1) * .0035)
                    self.assertAlmostEqual(scored['target_time_s'], (issued + 1) * .0035 + .05)
                    self.assertEqual(scored['horizon_s'], .05)
                    self.assertEqual(scored['raw_outputs'], original['raw'])
                    np.testing.assert_allclose(scored['quaternion_wxyz'], original['quaternion'], atol=1e-12)
                    self.assertLess(angle(scored['truth_wxyz'], truth), 1e-5)
                    self.assertAlmostEqual(scored['error_deg'], angle(original['quaternion'], truth), places=6)
                    for name, prediction in original['baselines'].items():
                        self.assertLess(angle(scored['baselines'][name]['quaternion_wxyz'], prediction), 1e-5)
                        self.assertAlmostEqual(scored['baselines'][name]['error_deg'], angle(prediction, truth), places=6)
                self.assertEqual(first_pending['status'], 'pending')
                self.assertIsNone(first_pending['truth_wxyz'])
                self.assertEqual(trace.read_text().splitlines().count('request STEP'), 20)

    def test_verified_reset_discards_pending_and_scored_forecasts_before_new_steps(self):
        actions = [None] + [self.target(i) for i in range(17)] + [None] + [self.target(i) for i in range(2)]
        requests, replies, _ = self.stream(actions)
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory), requests, b''.join(replies), module='host.motion_demo',
                                    export_dir=self.export) as (base, _):
                self.assertEqual(self.post(base, '/api/connect')[0], 200)
                for target in actions[1:18]:
                    code, state = self.post(base, '/api/step', {'target_wxyz': target})
                    self.assertEqual(code, 200, state)
                self.assertIsNotNone(state['sample'].get('scored_forecast'), 'Fixture must mature a forecast before RESET')
                code, reset = self.post(base, '/api/reset')
                self.assertEqual(code, 200, reset)
                self.assertTrue(reset['reset_verified'])
                self.assertEqual(reset['count_verified'], 0)
                self.assertIsNone(reset['sample'])
                self.assertEqual(reset['pending_forecast_count'], 0)
                for index, target in enumerate(actions[19:]):
                    code, state = self.post(base, '/api/step', {'target_wxyz': target})
                    self.assertEqual(code, 200, state)
                    self.assertEqual(state['sample']['index'], index)
                    self.assertEqual(state['pending_forecast_count'], index + 1)
                    self.assertIsNone(state['sample']['scored_forecast'])

    def test_invalid_sixteenth_usb_reply_preserves_pending_forecast_without_inventing_truth(self):
        actions = [None] + [self.target(i) for i in range(16)]
        requests, replies, _ = self.stream(actions)
        replies[-1] = replies[-1][:-1] + bytes([replies[-1][-1] ^ 1])
        with tempfile.TemporaryDirectory() as directory:
            with rotated_native_app(Path(directory), requests, b''.join(replies), module='host.motion_demo',
                                    export_dir=self.export) as (base, trace):
                self.assertEqual(self.post(base, '/api/connect')[0], 200)
                for target in actions[1:16]:
                    code, last = self.post(base, '/api/step', {'target_wxyz': target})
                    self.assertEqual(code, 200, last)
                self.assertIn('scored_forecast', last['sample'], 'Pending output must declare absence of matured truth')
                self.assertIsNone(last['sample']['scored_forecast'])
                code, failed = self.post(base, '/api/step', {'target_wxyz': actions[-1]})
                self.assertEqual(code, 502, failed)
                self.assertEqual(failed['status'], 'error')
                self.assertEqual(failed['count_verified'], 15)
                self.assertEqual(failed['pending_forecast_count'], 15)
                self.assertEqual(failed['sample'], last['sample'])
                self.assertEqual(trace.read_text().splitlines()[-2:], ['close', 'free'])
