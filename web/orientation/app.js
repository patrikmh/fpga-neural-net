import * as THREE from './vendor/three.module.js';
import {createOrientationScene} from './scene.js';

const element = id => document.getElementById(id);
const target = new THREE.Quaternion();
const sensorNames = ['aₓ', 'aᵧ', 'a𝓏', 'ωₓ', 'ωᵧ', 'ω𝓏', 'mₓ', 'mᵧ', 'm𝓏'];
const sensorColors = ['#5be6e9', '#eac279', '#869dff'];
let running = false, connected = false, busy = false, loopActive = false;
let pendingRequest = Promise.resolve(), resetPending = false;
let lastSample = null, lastSignature = null, selectedState = 0;
let motionMode = false, forecastWindow = [], lastScoredIndex = -1, lastMatured = null;
let previousHidden = null, deltas = Array(96).fill(null);
let stateHistory = Array.from({length: 96}, () => []);
let sensorHistory = Array.from({length: 9}, () => []);

const stateButtons = Array.from({length: 96}, (_, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'state-cell';
  button.textContent = `h${index}`;
  button.setAttribute('aria-label', `h${index}`);
  button.setAttribute('aria-pressed', String(index === 0));
  button.addEventListener('click', () => selectState(index));
  element('states').append(button);
  return button;
});
const sensorCards = sensorNames.map((name, index) => {
  const card = document.createElement('div');
  card.className = 'sensor-channel';
  const unit = index < 3 ? 'm/s²' : index < 6 ? 'rad/s' : 'µT';
  card.innerHTML = `<div class="sensor-title"><span>${name}</span><span>${unit}</span></div><output>—</output><canvas aria-label="Sensorhistorik ${name}, ${unit}"></canvas>`;
  element('sensor-traces').append(card);
  return card;
});
const canvas = element('scene');
let view, sceneError = '';
try {
  view = createOrientationScene(canvas, selectState);
} catch (error) {
  element('error').hidden = false;
  sceneError = `3D-vyn kräver WebGL: ${error.message}`;
  element('error').textContent = sceneError;
}

function remember(history, value) {
  history.push(value);
  if (history.length > 96) history.shift();
}
function trace(canvas, values, color) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.strokeStyle = '#223641'; context.lineWidth = 1;
  for (const fraction of [.2, .5, .8]) {
    context.beginPath(); context.moveTo(0, height * fraction); context.lineTo(width, height * fraction); context.stroke();
  }
  if (!values.length) return;
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const span = Math.max(maximum - minimum, .001);
  context.strokeStyle = color; context.lineWidth = 1.5;
  context.beginPath();
  values.forEach((value, index) => {
    const x = index / Math.max(1, values.length - 1) * width;
    const y = height * .82 - (value - minimum + (span - maximum + minimum) / 2) / span * height * .64;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
}
function selectedDetail() {
  const raw = lastSample?.hidden[selectedState];
  element('state-name').textContent = `h${selectedState}`;
  element('state-raw').textContent = raw === undefined ? '—' : String(raw);
  element('state-value').textContent = raw === undefined ? '—' : (raw / 8192).toFixed(5);
  const delta = deltas[selectedState];
  element('state-delta').textContent = delta === null ? '—' : `${delta > 0 ? '+' : ''}${delta}`;
  element('state-delta').title = 'Förändring i råa Q13-enheter sedan föregående verifierade steg';
  element('state-history-label').textContent = `Historik · ${stateHistory[selectedState].length} senaste verifierade steg`;
  trace(element('state-history'), stateHistory[selectedState], raw < 0 ? '#fa78c0' : '#5be6e9');
}
function selectState(index) {
  stateButtons[selectedState].setAttribute('aria-pressed', 'false');
  selectedState = index;
  stateButtons[index].setAttribute('aria-pressed', 'true');
  view?.select(index);
  selectedDetail();
}
function metric(id, value, unit = '') {
  const suffix = document.createElement('i'); suffix.textContent = unit;
  element(id).replaceChildren(document.createTextNode(value), suffix);
}
function clearForecasts() {
  element('forecast-status').textContent = motionMode ? 'Väntar på framtida verklig pose' : 'Hela vinkeln 0–180°, utan kapning';
  forecastWindow = []; lastScoredIndex = -1; lastMatured = null;
  for (const id of ['forecast-current-time', 'forecast-target-time', 'forecast-scored-time',
    'forecast-rmse', 'baseline-hold-error', 'baseline-cv-error', 'baseline-hold-rmse', 'baseline-cv-rmse']) {
    element(id).textContent = '—';
  }
  element('forecast-score-count').textContent = '0';
  element('forecast-pending-count').textContent = '0';
}
function showForecast(sample, pendingCount) {
  const forecast = sample.forecast;
  element('forecast-current-time').textContent = `${forecast.issued_at_s.toFixed(4)} s`;
  element('forecast-target-time').textContent = `${forecast.target_time_s.toFixed(4)} s`;
  element('forecast-pending-count').textContent = String(pendingCount);
  const scored = sample.scored_forecast;
  if (scored && scored.issued_index > lastScoredIndex) {
    lastScoredIndex = scored.issued_index;
    lastMatured = scored;
    forecastWindow.push(scored);
    if (forecastWindow.length > 96) forecastWindow.shift();
  }
  if (!lastMatured) return;
  metric('error-value', lastMatured.error_deg.toFixed(1), '°');
  element('forecast-scored-time').textContent = `${lastMatured.target_time_s.toFixed(4)} s`;
  element('forecast-status').textContent = `Äldre prognos · facit vid ${lastMatured.target_time_s.toFixed(4)} s`;
  element('forecast-score-count').textContent = String(forecastWindow.length);
  const methods = [['forecast', null], ['baseline-hold', 'sensor_hold'], ['baseline-cv', 'sensor_constant_velocity']];
  for (const [id, baseline] of methods) {
    const error = row => baseline ? row.baselines[baseline].error_deg : row.error_deg;
    const rmse = Math.sqrt(forecastWindow.reduce((sum, row) => sum + error(row) ** 2, 0) / forecastWindow.length);
    element(`${id}-rmse`).textContent = `${rmse.toFixed(1)}°`;
    if (baseline) element(`${id}-error`).textContent = `${error(lastMatured).toFixed(1)}°`;
  }
}
function clearMeasurements() {
  clearForecasts();
  lastSample = null; lastSignature = null; previousHidden = null;
  deltas = Array(96).fill(null);
  stateHistory = Array.from({length: 96}, () => []);
  sensorHistory = Array.from({length: 9}, () => []);
  stateButtons.forEach(button => { button.style.background = ''; button.style.color = ''; button.title = 'Inget verifierat värde'; });
  sensorCards.forEach(card => { card.querySelector('output').textContent = '—'; trace(card.querySelector('canvas'), [], '#5be6e9'); });
  for (const [id, unit] of [['error-value', '°'], ['model-time', 's'], ['compute-time', 'ms'], ['roundtrip', 'ms']]) metric(id, '—', unit);
  element('cycles').textContent = 'Väntar på cykelräknare';
  element('quaternion').textContent = 'FPGA wxyz · —';
  element('estimate-empty').hidden = false;
  view?.clear(); selectedDetail();
}
function measuredSample(sample, count, animate, pendingCount = 0) {
  const signature = `${count}:${sample.index}`;
  const fresh = signature !== lastSignature;
  if (fresh) {
    sample.hidden.forEach((raw, index) => {
      deltas[index] = previousHidden === null ? null : raw - previousHidden[index];
      remember(stateHistory[index], raw / 8192);
      const magnitude = Math.min(1, Math.abs(raw / 8192));
      stateButtons[index].style.background = raw < 0 ? `rgba(250,120,192,${.08 + magnitude * .4})` : `rgba(91,230,233,${.08 + magnitude * .4})`;
      stateButtons[index].style.color = raw < 0 ? '#fac0e1' : '#a6f4f3';
      stateButtons[index].title = `h${index}: rå Q13 ${raw}, värde ${(raw / 8192).toFixed(5)}`;
    });
    sample.sensors.forEach((value, index) => {
      remember(sensorHistory[index], value);
      sensorCards[index].querySelector('output').textContent = value.toFixed(3);
      trace(sensorCards[index].querySelector('canvas'), sensorHistory[index], sensorColors[Math.floor(index / 3)]);
    });
    previousHidden = [...sample.hidden]; lastSignature = signature;
  }
  lastSample = sample;
  metric('error-value', (sample.forecast?.error_deg ?? sample.error_deg)?.toFixed(1) ?? '—', '°');
  if (sample.forecast) {
    element('forecast-status').textContent = 'Väntar på framtida verklig pose';
    showForecast(sample, pendingCount);
  }
  metric('model-time', sample.model_time_s.toFixed(3), 's');
  metric('compute-time', (sample.compute_us / 1000).toFixed(3), 'ms');
  metric('roundtrip', sample.roundtrip_ms.toFixed(1), 'ms');
  element('cycles').textContent = `${sample.cycles.toLocaleString('sv-SE')} cykler · 50 MHz`;
  element('quaternion').textContent = `FPGA wxyz · ${sample.quaternion_wxyz.map(value => value.toFixed(3)).join(' / ')}`;
  element('estimate-empty').hidden = true;
  view?.sample(sample, fresh && animate); selectedDetail();
}
function controls() {
  element('connect').disabled = busy || connected;
  element('step').disabled = busy || running || resetPending || !connected;
  element('disconnect').disabled = busy || resetPending || !connected;
  element('run').disabled = !connected || resetPending || (busy && !loopActive);
  element('reset').disabled = !connected || resetPending;
  element('run').textContent = running ? 'Pausa' : 'Kör';
  if (connected) {
    element('flow-status').textContent = running ? 'LIVE · nya verifierade steg driver flödet' : 'Pausad · dra för att ändra nästa rörelsemål';
    element('stream-state').textContent = running ? 'Verifierad FPGA via USB' : 'Senast verifierat · pausad';
  }
}
function targetChanged() {
  target.normalize();
  const angles = new THREE.Euler().setFromQuaternion(target, 'ZYX');
  const degrees = value => (value * 180 / Math.PI).toFixed(0);
  element('target').textContent = `Mål · roll X ${degrees(angles.x)}° / pitch Y ${degrees(angles.y)}° / yaw Z ${degrees(angles.z)}°`;
  view?.setTarget([target.w, target.x, target.y, target.z]);
}
function rotateTarget(yaw, pitch, roll = 0) {
  target.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), yaw));
  target.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), pitch));
  target.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), roll));
  targetChanged();
}
let pointer = null, dragged = false;
canvas.addEventListener('pointerdown', event => {
  pointer = [event.clientX, event.clientY]; dragged = false;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!pointer) return;
  const x = event.clientX - pointer[0], y = event.clientY - pointer[1];
  if (Math.abs(x) + Math.abs(y) > 2) dragged = true;
  rotateTarget(x * .008, y * .008);
  pointer = [event.clientX, event.clientY];
});
canvas.addEventListener('pointerup', event => { if (!dragged) view?.pick(event); pointer = null; });
for (const name of ['pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { pointer = null; });
window.addEventListener('keydown', event => {
  if (event.target.closest('input,button,textarea,select,[contenteditable]')) return;
  const rotations = {ArrowLeft: [.12, 0, 0], ArrowRight: [-.12, 0, 0],
    ArrowUp: [0, .12, 0], ArrowDown: [0, -.12, 0], q: [0, 0, .12], e: [0, 0, -.12]};
  const rotation = rotations[event.key] || rotations[event.key.toLowerCase()];
  if (rotation) { event.preventDefault(); rotateTarget(...rotation); }
  if (event.key.toLowerCase() === 'r') { target.identity(); targetChanged(); }
});
function show(state, animate = false) {
  motionMode = state.task?.kind === 'motion_prediction';
  element('forecast-panel').hidden = !motionMode;
  if (motionMode) {
    const horizon = state.task.horizon_s * 1000;
    document.title = 'Neuro / Rörelseprognos på FPGA';
    element('output-label').textContent = `Prognos +${horizon} ms`;
    element('estimate-empty').textContent = 'Ingen verifierad rörelseprognos ännu';
    element('error-label').textContent = `PROGNOSFEL +${horizon} ms`;
    element('task-explanation').textContent = `En CfC-cell uppdaterar sitt inre tillstånd från sensorerna och föregående tillstånd. Den här modellen förutsäger kroppens orientering ${horizon} ms framåt i modelltid. Demonstrationen börjar upprätt efter verifierad återställning. Horisonten gäller modelltid, inte väggtid. Framtida abrupta muskommandon är okända och kan inte förutsägas. Verklig framtida pose och prognosfel inväntas tills rörelsen har inträffat.`;
    canvas.setAttribute('aria-label', `Dra för att styra rörelsemålet. Sensorpod, nio indata, en CfC-krets och en rörelseprognos ${horizon} ms framåt visas.`);
  }
  connected = ['connected', 'running'].includes(state.status);
  const labels = {disconnected: 'Frånkopplad', connected: 'Ansluten', running: 'Ansluten', error: 'USB-fel · stoppad'};
  element('status').textContent = labels[state.status] || state.status;
  element('status-dot').className = `status-dot ${state.status === 'error' ? 'error' : connected ? 'connected' : ''}`;
  element('error').hidden = !(state.error || sceneError);
  element('error').textContent = state.error || sceneError;
  if (state.status === 'disconnected' || (state.reset_verified && state.count_verified === 0)) {
    clearMeasurements();
    target.identity(); targetChanged();
  } else if (state.sample?.verified) {
    measuredSample(state.sample, state.count_verified, animate && state.status !== 'error', state.pending_forecast_count);
  }
  metric('count', motionMode || connected || state.status === 'error' ? String(state.count_verified) : '—');
  if (state.status === 'disconnected') {
    running = false; view?.pause();
    element('stream-state').textContent = 'Inga aktiva mätvärden';
    element('flow-status').textContent = 'Anslut FPGA:n för att börja · inga syntetiska utdata';
  }
  if (state.status === 'error') {
    running = false; view?.pause();
    element('stream-state').textContent = 'Senast verifierat · stoppad vid fel';
    element('flow-status').textContent = 'STOPPAD · senaste verifierade data visas';
  }
  controls();
}
async function api(path, payload = null) {
  if (busy) throw new Error('Ett FPGA-steg pågår redan.');
  busy = true; controls();
  try {
    pendingRequest = fetch(`/api/${path}`, payload === null ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    }).then(async response => {
      const state = await response.json();
      if ('count_verified' in state) show(state, path === 'step');
      if (path === 'state' && connected && state.target_wxyz) {
        const [w, x, y, z] = state.target_wxyz;
        target.set(x, y, z, w); targetChanged();
      }
      if (!response.ok) throw new Error(state.error || `HTTP ${response.status}`);
      return state;
    });
    return await pendingRequest;
  } finally { busy = false; controls(); }
}
function showError(error) {
  running = false; view?.pause();
  element('error').hidden = false; element('error').textContent = error.message;
  controls();
}
function step() { return api('step', {target_wxyz: [target.w, target.x, target.y, target.z]}); }
element('connect').addEventListener('click', async () => {
  target.identity(); targetChanged();
  try { await api('connect', {}); } catch (error) { showError(error); }
});
element('step').addEventListener('click', async () => {
  try { await step(); } catch (error) { showError(error); }
});
element('disconnect').addEventListener('click', async () => {
  running = false; view?.pause();
  try { await api('disconnect', {}); } catch (error) { showError(error); }
});
element('reset').addEventListener('click', async () => {
  running = false; resetPending = true; view?.pause(); controls();
  try {
    await pendingRequest;
    await api('reset', {});
    target.identity(); targetChanged();
  } catch (error) { showError(error); }
  finally { resetPending = false; controls(); }
});
element('run').addEventListener('click', async () => {
  running = !running; controls();
  if (!running) { view?.pause(); return; }
  if (loopActive) return;
  loopActive = true;
  try { while (running && connected) await step(); }
  catch (error) { showError(error); }
  finally { running = false; loopActive = false; controls(); }
});
element('blend').addEventListener('input', event => {
  const alpha = Number(event.target.value) / 100;
  const value = -.65 + 1.5 * alpha;
  element('blend-alpha').textContent = `α = ${alpha.toFixed(2).replace('.', ',')}`;
  element('blend-result').textContent = `h = ${value >= 0 ? '+' : ''}${value.toFixed(3).replace('.', ',')}`;
});
window.addEventListener('resize', () => {
  selectedDetail();
  sensorCards.forEach((card, index) => trace(card.querySelector('canvas'), sensorHistory[index], sensorColors[Math.floor(index / 3)]));
});
clearMeasurements(); controls(); targetChanged();
api('state').catch(showError);
