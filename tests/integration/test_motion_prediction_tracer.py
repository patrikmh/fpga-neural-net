"""TRACER: train a future-pose model and consume its USB output in a real browser.

Behavior: a generated 50 ms prediction task survives training, integer export,
native FTDI framing, HTTP and the browser without reporting current-pose error.
Only the external FPGA/USB device is replaced by the native libftdi fixture.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from model.broad_integer import GoldenBroadPredictor
from test_orientation_sim import reference_sensors, unit
from integration.test_orientation_demo import ROOT, crc_frame, rotated_native_app


TASK = {'kind': 'motion_prediction', 'horizon_s': .05, 'model_dt_s': .0035,
        'target': 'absolute_body_to_enu_wxyz'}


class MotionPredictionTracerTests(unittest.TestCase):
    def test_generated_future_pose_model_reaches_browser_as_verified_pending_forecast(self):
        python = str(ROOT / '.venv/bin/python')
        playwright = shutil.which('playwright-cli') or '/opt/homebrew/bin/playwright-cli'

        def command(arguments, timeout=90):
            result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
            self.assertEqual(result.returncode, 0,
                             f"Required pipeline command failed: {arguments}\n{result.stdout}{result.stderr}")
            return result.stdout + result.stderr

        with tempfile.TemporaryDirectory(prefix='motion-tracer-') as directory:
            workspace = Path(directory)
            data, training, export = (workspace / name for name in ('data', 'training', 'export'))
            command([python, '-m', 'model.motion_generate', '--output-dir', str(data), '--seed', '17',
                     '--train-sequences', '2', '--validation-sequences', '1', '--length', '64',
                     '--horizon-ms', '50'])
            command([python, '-m', 'model.motion_train', '--dataset-dir', str(data),
                     '--output-dir', str(training), '--epochs', '1', '--hidden-size', '96',
                     '--warmup', '4', '--chunk-size', '32', '--batch-size', '2', '--seed', '23',
                     '--learning-rate', '.001'])
            command([python, '-m', 'model.broad_export_integer', '--checkpoint',
                     str(training / 'checkpoint.pt'), '--output-dir', str(export)])
            manifest = json.loads((export / 'manifest.json').read_text())
            self.assertEqual(manifest['config']['task'], TASK)

            sensors = reference_sensors(unit([1, 0, 0, 0]), unit([1, 0, 0, 0]))
            normalized = [(float(value) - mean) / std for value, mean, std in
                          zip(sensors, manifest['normalization']['mean'], manifest['normalization']['std'])]
            inputs = [min(32767, max(-32768, math.floor(value * 2048 + .5) if value >= 0
                                    else -math.floor(-value * 2048 + .5))) for value in normalized]
            hidden, raw = GoldenBroadPredictor.from_export(export).step(inputs)
            norm = math.sqrt(sum(value * value for value in raw))
            self.assertGreater(norm, 0, 'Fixture must contain an actual nonzero model forecast')
            expected = {'task': TASK, 'hidden': hidden, 'raw': raw,
                        'quaternion': [value / norm for value in raw]}
            requests = crc_frame(struct.pack('<BBH9h', 0xa6, 2, 65535, *([0] * 9)))
            requests += crc_frame(struct.pack('<BBH9h', 0xa6, 1, 0, *inputs))
            replies = crc_frame(struct.pack('<BBH96h4hI', 0xa6, 0, 65535, *([0] * 96), *([0] * 4), 0))
            replies += crc_frame(struct.pack('<BBH96h4hI', 0xa6, 0, 0, *hidden, *raw, 124046))

            with rotated_native_app(workspace, requests, replies, module='host.motion_demo',
                                    export_dir=export) as (base, trace):
                session = 'm' + hashlib.sha256(str(workspace).encode()).hexdigest()[:8]
                script = """async page => {
                  const expected = EXPECTED;
                  const check = (ok, message) => { if (!ok) throw new Error(message); };
                  const errors = [];
                  page.on('pageerror', error => errors.push(error.message));
                  const post = async (name, path) => {
                    const reply = page.waitForResponse(response => response.url().endsWith(path)
                      && response.request().method() === 'POST');
                    await page.getByRole('button', {name}).click();
                    const response = await reply;
                    check(response.ok(), `HTTP ${response.status()} for ${path}`);
                    return response.json();
                  };
                  await post(/^(connect|anslut)$/i, '/api/connect');
                  try {
                    const state = await post(/^(step|steg)$/i, '/api/step');
                    check(Object.entries(expected.task).every(([key, value]) => state.task?.[key] === value),
                      'Task identity lost');
                    const sample = state.sample;
                    check(state.source === 'fpga_usb' && sample.verified, 'Forecast must be verified USB output');
                    check(JSON.stringify(sample.hidden) === JSON.stringify(expected.hidden), 'Hidden states differ from device');
                    check(JSON.stringify(sample.raw_outputs) === JSON.stringify(expected.raw), 'Raw forecast differs from device');
                    check(sample.quaternion_wxyz.every((v, i) => Math.abs(v - expected.quaternion[i]) < 1e-10),
                      'Rendered quaternion must come from actual chip outputs');
                    check(!('error_deg' in sample), 'Current-pose error must not masquerade as forecast error');
                    const f = sample.forecast;
                    check(f.status === 'pending' && f.truth_wxyz === null && f.error_deg === null,
                      'Future truth must stay pending until it exists');
                    check(Math.abs(f.issued_at_s - .0035) < 1e-12 && Math.abs(f.target_time_s - .0535) < 1e-12
                      && f.horizon_s === .05, 'Forecast must refer to exactly 50 ms ahead in model time');
                    const states = page.getByRole('region', {name: /measured states|uppmätta tillstånd/i});
                    check(await states.getByRole('button', {name: /^h95$/i}).count() === 1, 'All 96 states must remain accessible');
                    await page.getByText(/50\\s*ms/i).first().waitFor();
                    await page.getByText(/pending|väntar|inväntar/i).first().waitFor();
                    check(errors.length === 0, `Browser errors: ${errors.join('; ')}`);
                    return {motion_prediction_tracer: 'passed', horizon_ms: 50, states: 96};
                  } finally {
                    await post(/^(disconnect|koppla från)$/i, '/api/disconnect');
                  }
                }""".replace('EXPECTED', json.dumps(expected))
                try:
                    opened = command([playwright, f'-s={session}', 'open', base])
                    self.assertNotIn('### Error', opened, opened)
                    output = command([playwright, f'-s={session}', 'run-code', script])
                    self.assertNotIn('### Error', output, output)
                    self.assertRegex(output, r'"motion_prediction_tracer"\s*:\s*"passed"', output)
                finally:
                    subprocess.run([playwright, f'-s={session}', 'close'], cwd=ROOT,
                                   capture_output=True, text=True, timeout=15)
                events = trace.read_text().splitlines()
                self.assertEqual(events.count('request STEP'), 1)
                self.assertFalse(any(event.startswith('ERROR') for event in events), events)
