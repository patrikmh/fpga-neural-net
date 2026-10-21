// Run against a freshly disconnected host.motion_demo with its matching SRAM model.
// playwright-cli -s=motion-e2e run-code "$(cat tests/browser/motion-e2e.js)"
async page => {
  const check = (ok, message) => { if (!ok) throw new Error(message); };
  const errors = [];
  const capture = error => errors.push(error.message);
  page.on('pageerror', capture);
  const state = () => page.evaluate(async () => (await fetch('/api/state')).json());
  const control = name => page.getByRole('button', {name});
  const post = async (name, path) => {
    const pending = page.waitForResponse(response => response.url().endsWith(`/api/${path}`)
      && response.request().method() === 'POST');
    await control(name).click();
    const response = await pending;
    check(response.ok(), `${path} HTTP ${response.status()}`);
    const result = await response.json();
    await page.waitForFunction(count => document.querySelector('#count').textContent.trim() === String(count),
      result.count_verified);
    return result;
  };
  const current = await state();
  check(current.task?.kind === 'motion_prediction', 'Requires the motion server and a real motion export');
  check(current.status === 'disconnected', 'Start with a fresh disconnected session');
  const horizon = current.task.horizon_s;
  let connected = false;
  try {
    const initial = await post(/^anslut$|^connect$/i, 'connect');
    connected = true;
    check(initial.reset_verified && initial.count_verified === 0, 'Connect must verify a clean board reset');
    const issued = new Map();
    let measured;
    for (let index = 1; index <= 40; index++) {
      await page.locator('#scene').focus();
      await page.keyboard.press(index < 25 ? 'ArrowRight' : 'ArrowUp');
      measured = await post(/^steg$|^step$/i, 'step');
      const sample = measured.sample;
      check(sample.verified && sample.hidden.length === 96 && sample.raw_outputs.length === 4,
        'Each displayed forecast must have a verified full-state FPGA reply');
      check(sample.forecast.error_deg === null && sample.forecast.truth_wxyz === null,
        'A newly issued forecast cannot already know future truth');
      check(Math.abs(sample.forecast.target_time_s - sample.model_time_s - horizon) < 1e-10,
        'Forecast target must use model time');
      issued.set(sample.index, sample.forecast);
      if (index < 16) check(sample.scored_forecast === null, 'No score before the future target has elapsed');
      if (sample.scored_forecast) {
        const scored = sample.scored_forecast;
        check(JSON.stringify(scored.quaternion_wxyz) === JSON.stringify(issued.get(scored.issued_index).quaternion_wxyz),
          'Scoring must retain the immutable originally issued prediction');
        check(scored.target_time_s <= sample.model_time_s && scored.issued_index < sample.index,
          'The score must belong to an older matured forecast');
        const displayedTime = await page.locator('#forecast-comparison-time').innerText();
        check(displayedTime.includes(scored.target_time_s.toFixed(4)), 'Both comparison objects need the same target time');
      }
    }
    check(measured.sample.sensors.slice(3, 6).some(value => Math.abs(value) > .1),
      'Exercise actual simulated movement, not only stationary forecasts');
    await control(/^kör$|^run$/i).click();
    await page.waitForFunction(count => Number(document.querySelector('#count').textContent) > count + 4,
      measured.count_verified);
    await control(/^pausa$|^pause$/i).click();
    await page.waitForFunction(() => !document.querySelector('#step').disabled);
    measured = await state();
    check(measured.count_verified >= 45, 'Run must produce at least five additional verified FPGA steps');
    await page.waitForTimeout(300);
    check((await state()).count_verified === measured.count_verified, 'Pause must stop further model steps');
    const result = {verified_steps: measured.count_verified, hidden_states: 96,
      horizon_s: horizon, matched_forecasts: await page.locator('#forecast-score-count').innerText(),
      rolling_model_rmse: await page.locator('#forecast-rmse').innerText(),
      rolling_cv_rmse: await page.locator('#baseline-cv-rmse').innerText(),
      quality_claim: 'interactive demonstration only; not held-out qualification'};
    const reset = await post(/^återställ$|^reset$/i, 'reset');
    check(reset.reset_verified && reset.sample === null && reset.pending_forecast_count === 0,
      'Acknowledged reset must clear pending and scored forecasts');
    check(await page.locator('#forecast-score-count').innerText() === '0', 'Reset must clear rolling scores');
    check(errors.length === 0, `Browser errors: ${errors.join('; ')}`);
    return result;
  } finally {
    if (connected) {
      const pause = control(/^pausa$|^pause$/i);
      if (await pause.count()) await pause.click();
      await page.waitForFunction(() => !document.querySelector('#disconnect').disabled);
      await post(/^koppla från$|^disconnect$/i, 'disconnect');
    }
    page.removeListener('pageerror', capture);
  }
}
