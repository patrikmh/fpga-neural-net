"""TRACER: browser presentation separates pending forecasts from matured scores.

Behavior: verified motion samples produce a fair matched-window comparison in
the browser, remain stable on reads, and clear after acknowledged hardware RESET.
Only the external FPGA/USB device is replaced by the existing native fixture.
"""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from integration import test_motion_demo as motion_fixture
from integration.test_orientation_demo import ROOT, rotated_native_app


class MotionPresentationTests(unittest.TestCase):
    def test_browser_separates_pending_forecast_from_matched_matured_metrics_and_resets(self):
        motion_fixture.MotionDemoTests.setUpClass()
        try:
            fixture = motion_fixture.MotionDemoTests()
            actions = [None] + [[1., 0., 0., 0.]] * 17 + [None]
            requests, replies, _ = fixture.stream(actions)
            with tempfile.TemporaryDirectory(prefix='motion-presentation-') as temporary:
                workspace = Path(temporary)
                with rotated_native_app(workspace, requests, b''.join(replies), module='host.motion_demo',
                                        export_dir=fixture.export) as (base, trace):
                    playwright = shutil.which('playwright-cli') or '/opt/homebrew/bin/playwright-cli'
                    session = 'p' + hashlib.sha256(str(workspace).encode()).hexdigest()[:8]
                    script = r"""async page => {
                      const check = (ok, message) => { if (!ok) throw new Error(message); };
                      const errors = [];
                      page.on('pageerror', error => errors.push(error.message));
                      const text = id => page.locator(`#${id}`).innerText();
                      const number = async id => Number((await text(id)).replace(',', '.').match(/-?\d+(?:\.\d+)?/)?.[0]);
                      const near = (a, b) => Math.abs(a - b) <= .11;
                      const post = async (name, path) => {
                        const completed = page.waitForResponse(response => response.url().endsWith(path)
                          && response.request().method() === 'POST');
                        await page.getByRole('button', {name}).click();
                        const response = await completed;
                        check(response.ok(), `${path} HTTP ${response.status()}`);
                        const state = await response.json();
                        await response.finished();
                        await page.waitForFunction(count => document.querySelector('#count')?.textContent.trim() === String(count), state.count_verified);
                        return state;
                      };
                      await page.setViewportSize({width: 390, height: 844});
                      check(await page.locator('#forecast-panel').isVisible(), 'Motion mode needs a visible forecast comparison panel');
                      const comparison = page.locator('#forecast-comparison');
                      check(await comparison.isVisible(), 'Motion mode needs a visible same-target-time 3D comparison');
                      check(await comparison.getAttribute('role') === 'region'
                        && /same|samma/i.test(await comparison.getAttribute('aria-label') || ''),
                        '3D comparison needs an accessible region naming its shared time');
                      check(await comparison.locator('canvas').count() === 1, 'Same-time comparison must include a real 3D canvas');
                      const checkQuaternion = async (id, expected) => {
                        const values = (await text(id)).replaceAll(',', '.').match(/[-+]?\d+(?:\.\d+)?/g)?.map(Number);
                        check(values?.length === 4 && values.every((value, index) => Math.abs(value - expected[index]) < 1e-6),
                          `${id} must expose the actual matched wxyz values`);
                      };
                      const explanation = await page.locator('#task-explanation').innerText();
                      check(/50\s*ms/i.test(explanation), 'Explain prediction horizon in model time');
                      check(/upprätt|upright|identity|identitet/i.test(explanation), 'Explain known upright reset domain');
                      check(/okänd|unknown|abrupt|oförutsäg/i.test(explanation), 'Explain that future abrupt mouse movement is unknown');
                      await post(/^(connect|anslut)$/i, '/api/connect');
                      let connected = true;
                      try {
                        const scored = [];
                        let state;
                        for (let step = 1; step <= 17; step++) {
                          state = await post(/^(step|steg)$/i, '/api/step');
                          check(state.sample.forecast.status === 'pending', 'Newest prediction remains pending');
                          const current = state.sample.forecast;
                          const currentTime = current.issued_at_s.toFixed(4);
                          const targetTime = current.target_time_s.toFixed(4);
                          check((await text('forecast-current-time')).replace(',', '.').includes(currentTime), 'Current issue timestamp must retain four decimal places');
                          check((await text('forecast-target-time')).replace(',', '.').includes(targetTime), 'Future target timestamp must retain four decimal places');
                          check(await number('forecast-pending-count') === Math.min(step, 15), 'Display actual pending queue count');
                          if (step <= 15) {
                            for (const id of ['forecast-issued-quaternion', 'forecast-observed-quaternion', 'forecast-comparison-time'])
                              check((await text(id)).includes('—'), `Before future truth arrives, ${id} must wait`);
                            check((await text('error-value')).includes('—'), 'No current-pose error may stand in for unavailable future truth');
                            check(await number('forecast-score-count') === 0, 'No scores before the horizon matures');
                            continue;
                          }
                          const matured = state.sample.scored_forecast;
                          scored.push(matured);
                          await checkQuaternion('forecast-issued-quaternion', matured.quaternion_wxyz);
                          await checkQuaternion('forecast-observed-quaternion', matured.truth_wxyz);
                          check((await text('forecast-comparison-time')).replace(',', '.').includes(matured.target_time_s.toFixed(4)),
                            'Both comparison objects must refer to the same exact matured target time');
                          check((await text('forecast-scored-time')).replace(',', '.').includes(matured.target_time_s.toFixed(4)),
                            'Older scored target needs its own exact timestamp');
                          check(near(await number('error-value'), matured.error_deg), 'Main error must refer to the older matured forecast');
                          check(near(await number('baseline-hold-error'), matured.baselines.sensor_hold.error_deg), 'Show hold baseline including zero');
                          check(near(await number('baseline-cv-error'), matured.baselines.sensor_constant_velocity.error_deg), 'Show constant-velocity baseline including zero');
                          check(await number('forecast-score-count') === scored.length, 'Scores are unique by issued index');
                          const rmse = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
                          check(near(await number('forecast-rmse'), rmse(scored.map(row => row.error_deg))), 'Neural RMSE uses matched matured records');
                          check(near(await number('baseline-hold-rmse'), rmse(scored.map(row => row.baselines.sensor_hold.error_deg))), 'Hold RMSE uses the same issued indices');
                          check(near(await number('baseline-cv-rmse'), rmse(scored.map(row => row.baselines.sensor_constant_velocity.error_deg))), 'CV RMSE uses the same issued indices');
                        }
                        check(scored.length === 2 && scored[0].issued_index === 0 && scored[1].issued_index === 1,
                          'Comparison window must contain exactly the first two matured forecasts');
                        check(scored.some(row => row.baselines.sensor_hold.error_deg < row.error_deg), 'Fixture must expose a visible baseline win');
                        const count = await number('forecast-score-count');
                        await page.evaluate(async () => { await fetch('/api/state'); await fetch('/api/state'); });
                        await page.waitForTimeout(250);
                        check(await number('forecast-score-count') === count, 'Read-only state fetches must not duplicate scoring');
                        check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile layout must not scroll horizontally');
                        for (const name of [/^(step|steg)$/i, /^(reset|återställ)$/i, /^(disconnect|koppla från)$/i]) {
                          const box = await page.getByRole('button', {name}).boundingBox();
                          check(box && box.width >= 44 && box.height >= 44, 'Mobile controls need 44px touch targets');
                        }
                        await post(/^(reset|återställ)$/i, '/api/reset');
                        const resetCaption = await text('forecast-status');
                        check(!/äldre prognos/i.test(resetCaption)
                          && !resetCaption.replace(',', '.').includes(scored[1].target_time_s.toFixed(4))
                          && /vänt|pending|ingen|reset|återställ/i.test(resetCaption),
                          'Verified RESET must clear the older scored timestamp and restore a waiting caption');
                        check(await number('forecast-score-count') === 0 && await number('forecast-pending-count') === 0,
                          'Verified RESET clears comparison window and pending count');
                        for (const id of ['error-value', 'forecast-rmse', 'baseline-hold-rmse', 'baseline-cv-rmse'])
                          check((await text(id)).includes('—'), `RESET clears ${id}`);
                        for (const id of ['forecast-issued-quaternion', 'forecast-observed-quaternion', 'forecast-comparison-time'])
                          check((await text(id)).includes('—'), `RESET clears matched comparison ${id}`);
                        check(errors.length === 0, `Browser errors: ${errors.join('; ')}`);
                        return {motion_presentation: 'passed', matched_scores: 2, mobile_width: 390};
                      } finally {
                        if (connected) await post(/^(disconnect|koppla från)$/i, '/api/disconnect');
                      }
                    }"""
                    def browser(*arguments):
                        result = subprocess.run([playwright, f'-s={session}', *arguments], cwd=ROOT,
                                                capture_output=True, text=True, timeout=90)
                        output = result.stdout + result.stderr
                        self.assertEqual(result.returncode, 0, output)
                        self.assertNotIn('### Error', output, output)
                        return output
                    try:
                        browser('open', base)
                        output = browser('run-code', script)
                        self.assertRegex(output, r'"motion_presentation"\s*:\s*"passed"')
                    finally:
                        subprocess.run([playwright, f'-s={session}', 'close'], cwd=ROOT,
                                       capture_output=True, text=True, timeout=15)
                    events = trace.read_text().splitlines()
                    self.assertEqual(events.count('request STEP'), 17)
                    self.assertEqual(events.count('request RESET'), 2)
                    self.assertEqual(events[-2:], ['close', 'free'])
        finally:
            motion_fixture.MotionDemoTests.tearDownClass()
