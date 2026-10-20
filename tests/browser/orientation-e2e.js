// playwright-cli -s=orientation-e2e run-code "$(cat tests/browser/orientation-e2e.js)"
// Append ?preflight=1 to the page URL for a read-only check without USB access.
// Behaviors: selectable measured states; illustration isolation; verified run/pause/reset.
async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const pageErrors = [];
  const captureError = error => pageErrors.push(error.message);
  page.on('pageerror', captureError);
  const clickPost = async (control, path) => {
    const completed = page.waitForResponse(response =>
      response.url().endsWith(`/api/${path}`) && response.request().method() === 'POST');
    await control.click();
    const response = await completed;
    check(response.ok(), `${path} HTTP ${response.status()}`);
    await response.finished();
    return response.json();
  };
  const state = () => page.evaluate(async () => {
    const response = await fetch('/api/state');
    if (!response.ok) throw new Error(`State HTTP ${response.status}`);
    return response.json();
  });
  const states = page.getByRole('region', {name: /measured states|uppmätta tillstånd/i});
  const detail = page.getByRole('region', {name: /measured state details|uppmätt tillståndsdetalj/i});
  const blend = page.getByRole('slider', {name: /illustrative blend|illustrativ blandning/i});
  const connect = page.getByRole('button', {name: /^(anslut|connect)$/i});
  const step = page.getByRole('button', {name: /^(steg|step)$/i});
  const run = page.getByRole('button', {name: /^(kör|run)$/i});
  const reset = page.getByRole('button', {name: /^(återställ|reset)$/i});
  const disconnect = page.getByRole('button', {name: /^(koppla från|disconnect)$/i});

  check(await states.count() === 1, 'Missing accessible region for the 96 measured FPGA states');
  for (let index = 0; index < 96; index++) {
    check(await states.getByRole('button', {name: new RegExp(`^h${index}$`, 'i')}).count() === 1,
      `Measured state h${index} must be individually selectable`);
  }
  check(await detail.count() === 1, 'Selected measured state needs an accessible detail region');
  check(await blend.count() === 1, 'Illustrative blend must have its own labeled slider');
  check(await reset.count() === 1, 'Verified hardware reset control is required');
  if (await page.evaluate(() => new URL(location.href).searchParams.has('preflight'))) {
    page.removeListener('pageerror', captureError);
    check(pageErrors.length === 0, `Browser errors: ${pageErrors.join('; ')}`);
    return {preflight: 'passed', usb: 'untouched'};
  }

  check((await state()).status === 'disconnected', 'Start with a fresh disconnected app');
  let connected = false;
  try {
    const connection = await clickPost(connect, 'connect');
    connected = true;
    check(connection.status === 'connected' && connection.reset_verified && connection.count_verified === 0,
      'Connect must complete with a fresh verified RESET and zero samples');
    let measured = await clickPost(step, 'step');
    check(measured.status === 'running' && measured.count_verified === 1 &&
      measured.sample?.verified && measured.sample.hidden.length === 96,
      'UI must consume a verified FPGA response containing all 96 states');

    await run.click();
    const initialCount = measured.count_verified;
    await page.waitForFunction(async count =>
      (await (await fetch('/api/state')).json()).count_verified > count + 1, initialCount);
    await page.getByRole('button', {name: /^(pausa|pause)$/i}).click();
    // A pause can have one request in flight. Wait for that request before comparing snapshots.
    await step.waitFor({state: 'visible'});
    await page.waitForFunction(() => [...document.querySelectorAll('button')]
      .some(button => /^(steg|step)$/i.test(button.textContent.trim()) && !button.disabled));
    measured = await state();
    await page.waitForTimeout(400);
    check((await state()).count_verified === measured.count_verified, 'Pause must settle without further FPGA writes');

    await states.getByRole('button', {name: /^h95$/i}).click();
    const text = await detail.innerText();
    check(text.includes(String(measured.sample.hidden[95])), 'Selected raw state must match the settled FPGA sample');
    check(/Q13/i.test(text) && /delta|Δ|förändring/i.test(text) && /history|historik/i.test(text),
      'Measured state details must expose Q13, change and history');
    const beforeIllustration = JSON.stringify(measured.sample);
    await blend.focus();
    await page.keyboard.press('Home');
    await page.keyboard.press('ArrowRight');
    check(JSON.stringify((await state()).sample) === beforeIllustration,
      'Changing an illustrative blend must not alter the measured FPGA sample');

    const cleared = await clickPost(reset, 'reset');
    check(cleared.reset_verified && cleared.count_verified === 0 && cleared.sample === null,
      'RESET must complete with verified empty measurements');
    check(pageErrors.length === 0, `Browser errors: ${pageErrors.join('; ')}`);
    return {verified_steps: measured.count_verified, states: 96, illustration: 'isolated', reset: 'verified'};
  } finally {
    if (connected) {
      // Pause any surviving run loop before disconnecting and releasing the real USB context.
      const pause = page.getByRole('button', {name: /^(pausa|pause)$/i});
      if (await pause.count()) await pause.click();
      await page.waitForFunction(() => [...document.querySelectorAll('button')]
        .some(button => /^(koppla från|disconnect)$/i.test(button.textContent.trim()) && !button.disabled));
      await clickPost(disconnect, 'disconnect');
    }
    page.removeListener('pageerror', captureError);
    check(pageErrors.length === 0, `Browser errors: ${pageErrors.join('; ')}`);
  }
}
