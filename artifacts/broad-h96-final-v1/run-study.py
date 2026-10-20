"""Execute the predeclared final trials after immutable selection/build freeze."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'tests')]
from model.broad_evaluate import current_identities, orientation_metrics
from model.broad_integer import GoldenBroadPredictor, quantize_inputs, sha256
from integration.test_broad_integer import independent_broad_step, full_rmse

plan_path = root / 'experiments/broad-h96-final-study-v1.json'
selection_path = root / 'artifacts/broad-h96-final-v1/selection.json'
plan = json.loads(plan_path.read_text())
selection = json.loads(selection_path.read_text())
selection_hash = sha256(selection_path)
checkpoint = root / plan['checkpoint']
export = root / plan['export_dir']


def verify_frozen():
    assert sha256(selection_path) == selection_hash
    assert sha256(plan_path) == selection['plan_sha256']
    assert sha256(Path(__file__)) == selection['recipe_sha256']
    assert sha256(checkpoint) == plan['checkpoint_sha256'] == selection['checkpoint_sha256']
    assert sha256(export / 'manifest.json') == selection['integer_export_manifest_sha256']
    sources, versions = current_identities()
    assert sources == selection['model_sources'] and versions == selection['versions']
    for name, value in selection['project_sources'].items():
        assert sha256(root / name) == value, name
    for field, path_field in (('build_report_sha256', 'build_report'),
                             ('bitstream_sha256', 'bitstream'),
                             ('rtl_report_sha256', 'rtl_report')):
        assert sha256(root / selection[path_field]) == selection[field]


verify_frozen()
predictor = GoldenBroadPredictor.from_export(export)
assert predictor.hidden_size == 96
assert predictor.checkpoint['prepared_manifest_sha256'] == plan['training_manifest_sha256']
build_report = json.loads((root / selection['build_report']).read_text())
rtl_report = json.loads((root / selection['rtl_report']).read_text())
assert build_report['timing']['achieved_mhz'] >= 50
assert build_report['export']['manifest_sha256'] == selection['integer_export_manifest_sha256']
assert build_report['bitstream']['sha256'] == selection['bitstream_sha256']
for name, value in build_report['sources'].items():
    assert selection['project_sources'][name] == value, name
assert build_report['sources']['rtl/broad_core.sv'] == selection['project_sources']['rtl/broad_core.sv']
assert rtl_report['sources']['rtl/broad_core.sv'] == selection['project_sources']['rtl/broad_core.sv']
assert rtl_report['result']['matched_steps'] == 10000 and rtl_report['result']['reset_replay']
rtl_stage_path = (root / selection['rtl_report']).parent / 'stage-manifest.json'
assert sha256(rtl_stage_path) == rtl_report['stage_manifest_sha256']
assert json.loads(rtl_stage_path.read_text())['export_manifest_sha256'] == selection['integer_export_manifest_sha256']
cycles = rtl_report['result']['cycles']
output = root / 'build/broad-h96-final-v1'
raw = root / 'build/datasets/broad-h96-final-v1'
output.mkdir(exist_ok=False)
raw.mkdir(exist_ok=False)
shutil.copy2(selection_path, output / 'selection.json')
(output / 'execution-freeze.json').write_text(json.dumps({
    'selection_sha256': sha256(selection_path), 'plan_sha256': sha256(plan_path),
    'recipe_sha256': sha256(Path(__file__)), 'raw_download_started_at_utc':
    datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2) + '\n')


def download(trial):
    name = trial['file']
    assert Path(name).name == name
    url = f"https://raw.githubusercontent.com/dlaidig/broad/{plan['source_revision']}/data_hdf5/{name}"
    destination = raw / name
    command = ['curl', '--fail', '--location', '--retry', '2', '--connect-timeout', '15',
               '--max-time', '120', '--output', str(destination), url]
    with (output / f"download-{trial['id']}.log").open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return dict(trial, sha256=sha256(destination), bytes=destination.stat().st_size, url=url)


print('Downloading four frozen new trials; no parsing yet', flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    downloads = list(pool.map(download, plan['trials']))
(output / 'download-manifest.json').write_text(json.dumps(downloads, indent=2) + '\n')
verify_frozen()
sources, versions = current_identities()
spec = {name: plan[name] for name in ('schema', 'repository', 'source_revision', 'license',
    'training_manifest_sha256', 'warmup', 'baseline_tau_s', 'heading_epsilon',
    'training_trial_ids', 'validation_trial_ids', 'scoring')}
spec.update(trials=[{key: entry[key] for key in ('id', 'file', 'split', 'rate_hz', 'sha256')}
                    for entry in downloads], checkpoint_sha256=plan['checkpoint_sha256'],
    selection_sha256=sha256(selection_path), sources=sources, versions=versions,
    integer_export_manifest_sha256=selection['integer_export_manifest_sha256'],
    build_report_sha256=selection['build_report_sha256'], rtl_report_sha256=selection['rtl_report_sha256'],
    frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selection_frozen_at_utc=selection['frozen_at_utc'],
    scope='One fixed held-out evaluation; raw hashes recorded before parsing; no test-driven tuning')
spec_path = output / 'test-spec.json'
spec_path.write_text(json.dumps(spec, indent=2) + '\n')
prepared, floating = output / 'prepared', output / 'float'
commands = [
    [str(root / 'build/broad-analysis-env/bin/python'), '-m', 'experiments.broad_prepare',
     '--data-dir', str(raw), '--spec', str(spec_path), '--split', 'test', '--output-dir', str(prepared)],
    [sys.executable, '-m', 'model.broad_evaluate', '--checkpoint', str(checkpoint),
     '--prepared-dir', str(prepared), '--spec', str(spec_path), '--output-dir', str(floating)],
]
for number, command in enumerate(commands):
    print('Starting prepare' if number == 0 else 'Starting full-trial float evaluation', flush=True)
    with (output / f'command-{number}.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
verify_frozen()
float_report = json.loads((floating / 'report.json').read_text())
manifest = predictor.manifest
mean, std = (np.asarray(manifest['normalization'][name], dtype=np.float64) for name in ('mean', 'std'))
weights, tanh, sigmoid = [] , [], []
for destination, name in ((weights, 'weights.hex'), (tanh, 'tanh.hex'), (sigmoid, 'sigmoid.hex')):
    for word in (export / name).read_text().split():
        value = int(word, 16)
        destination.append(value - (65536 if value >= 32768 else 0))
integer = output / 'integer'
integer.mkdir()
counter_names = ('input_clipping_count', 'state_clipping_count', 'raw_clipping_count',
                 'zero_quaternion_count', 'lut_clamp_count')
totals = {name: 0 for name in counter_names}
results, scalar_checks = [], 0
for entry in float_report['trials']:
    trial_id = entry['trial_id']
    with np.load(floating / entry['predictions_file'], allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in
                  ('sensors', 'targets', 'predictions', 'movement', 'reference_valid', 'sample_index')}
    sensors, targets = arrays['sensors'], arrays['targets']
    quantized, clips = quantize_inputs((sensors.astype(np.float64) - mean) / std)
    hidden = np.empty((len(sensors), 96), dtype=np.int16)
    raw_outputs = np.empty((len(sensors), 4), dtype=np.int16)
    predictor.reset()
    for index, values in enumerate(quantized):
        hidden[index], raw_outputs[index] = predictor.step(values)
    raw_values = raw_outputs.astype(np.float64) / 8192
    norms = np.linalg.norm(raw_values, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    predicted = raw_values / np.where(norms > 0, norms, 1)
    predicted[zero] = [1, 0, 0, 0]
    mask = (arrays['movement'] & arrays['reference_valid']).copy()
    mask[:plan['warmup']] = False
    measured = orientation_metrics(predicted, targets, mask, plan['heading_epsilon'])
    for kind, prediction in (('float', arrays['predictions']), ('integer', predicted)):
        wanted = entry['cfc']['full_rmse_deg'] if kind == 'float' else measured['full_rmse_deg']
        np.testing.assert_allclose(full_rmse(prediction, targets, mask), wanted, rtol=0, atol=1e-9)
    indexes = np.unique(np.r_[np.arange(4), np.random.default_rng(731 + trial_id).integers(0, len(sensors), 32)])
    for index in indexes:
        old = hidden[index - 1].tolist() if index else [0] * 96
        expected_h, expected_raw, *_ = independent_broad_step(quantized[index].tolist(), old, weights, tanh, sigmoid)
        np.testing.assert_array_equal(expected_h, hidden[index])
        np.testing.assert_array_equal(expected_raw, raw_outputs[index])
        scalar_checks += 1
    counters = dict(predictor.counters, input_clipping_count=clips, zero_quaternion_count=int(zero.sum()))
    for name in counter_names:
        totals[name] += counters[name]
    prediction_path = integer / f'{trial_id:02d}.npz'
    np.savez_compressed(prediction_path, sensors=sensors, targets=targets, quantized_inputs=quantized,
        hidden=hidden, raw_outputs=raw_outputs, integer_predictions=predicted,
        float_predictions=arrays['predictions'], movement=arrays['movement'],
        reference_valid=arrays['reference_valid'], sample_index=arrays['sample_index'])
    usb_dir = output / 'usb-vectors' / f'{trial_id:02d}'
    usb_dir.mkdir(parents=True)
    count = plan['physical_prefix_steps_per_trial']
    assert len(sensors) >= count
    vector_file = usb_dir / 'vectors.hex'
    with vector_file.open('x') as stream:
        for inputs, states, values in zip(quantized[:count], hidden[:count], raw_outputs[:count]):
            stream.write(''.join(f'{int(value) & 65535:04x}' for value in np.r_[inputs, states, values]) + '\n')
    vector_manifest = {'count': count, 'hidden_size': 96, 'initial_hidden': [0] * 96,
        'numeric': manifest['numeric'], 'export_manifest_sha256': sha256(export / 'manifest.json'),
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'vectors_sha256': sha256(vector_file),
        'expected_cycles': cycles, 'rtl_report_sha256': selection['rtl_report_sha256'],
        'rtl_core_sha256': selection['project_sources']['rtl/broad_core.sv'],
        'test_spec_sha256': sha256(spec_path), 'trial_id': trial_id}
    (usb_dir / 'manifest.json').write_text(json.dumps(vector_manifest, indent=2) + '\n')
    results.append({'trial_id': trial_id, 'float': entry['cfc'], 'integer': measured, **counters,
        'predictions_file': str(prediction_path.relative_to(output)),
        'predictions_sha256': sha256(prediction_path), 'usb_vectors_directory': str(usb_dir.relative_to(output)),
        'usb_vectors_manifest_sha256': sha256(usb_dir / 'manifest.json')})
    print(f"Trial {trial_id}: float/integer full RMSE {entry['cfc']['full_rmse_deg']:.6f}/{measured['full_rmse_deg']:.6f}", flush=True)
verify_frozen()
assert [entry['trial_id'] for entry in results] == [int(entry['id']) for entry in plan['trials']]
means = {}
for kind in ('float', 'integer'):
    means[kind] = {}
    for metric in ('full_rmse_deg', 'inclination_rmse_deg', 'heading_rmse_deg'):
        values = [entry[kind][metric] for entry in results if entry[kind][metric] is not None]
        means[kind][metric] = float(np.mean(values)) if values else None
report = {'split': 'new_test', 'test_used_for_selection': False, 'trials': results,
    'mean_trial_metrics': means, 'baselines': float_report['mean_trial_metrics']['baselines'], **totals,
    'integer_minus_float_full_rmse_deg': means['integer']['full_rmse_deg'] - means['float']['full_rmse_deg'],
    'independent_scalar_checked_steps': scalar_checks, 'selection_sha256': sha256(selection_path),
    'test_spec_sha256': sha256(spec_path), 'prepared_manifest_sha256': sha256(prepared / 'manifest.json'),
    'float_report_sha256': sha256(floating / 'report.json'),
    'checkpoint_sha256': manifest['checkpoint_sha256'], 'integer_export_manifest_sha256': sha256(export / 'manifest.json'),
    'recipe_sha256': sha256(Path(__file__)), 'physical_verification': 'not yet run',
    'sources': selection['model_sources'], 'project_sources': selection['project_sources'], 'versions': versions}
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
