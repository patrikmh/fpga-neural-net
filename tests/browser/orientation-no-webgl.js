// Open the actual app, then run with playwright-cli -s=orientation-no-webgl run-code.
// Boundary fault only: WebGL is unavailable; HTTP and 2D canvases remain real.
async page => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (kind, ...args) {
      if (['webgl', 'webgl2', 'experimental-webgl'].includes(kind)) return null;
      return original.call(this, kind, ...args);
    };
  });
  const initialState = page.waitForResponse(response =>
    response.url().endsWith('/api/state') && response.request().method() === 'GET');
  await page.reload();
  const response = await initialState;
  if (!response.ok()) throw new Error(`Initial state HTTP ${response.status()}`);
  await response.finished();
  // Let the state response render before checking that it preserves the warning.
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const alerts = page.getByRole('alert').filter({hasText: /WebGL/i});
  if (await alerts.count() !== 1 || !await alerts.isVisible()) {
    throw new Error('WebGL unavailable warning must remain visible after the initial state loads');
  }
  return {webgl_warning: 'visible', usb: 'untouched'};
}
