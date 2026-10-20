"""Freeze the selected, successfully routed H96 build before new data access."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
sys.path.insert(0, str(root))
from model.broad_evaluate import current_identities
from model.broad_integer import GoldenBroadPredictor, sha256

plan_path = root / 'experiments/broad-h96-final-study-v1.json'
plan = json.loads(plan_path.read_text())
build_path = root / 'artifacts/broad-capacity-v1/final/report.json'
build = json.loads(build_path.read_text())
rtl_path = root / 'artifacts/broad-rtl-v2/report.json'
rtl = json.loads(rtl_path.read_text())
bitstream = build_path.parent / build['bitstream']['file']
assert build['top'] == 'broad_top' and build['hidden_size'] == 96
assert build['timing']['constrained_mhz'] == 50 and build['timing']['achieved_mhz'] >= 50
assert build['utilization']['BSRAM']['used'] == 48
assert sha256(bitstream) == build['bitstream']['sha256']
predictor = GoldenBroadPredictor.from_export(root / plan['export_dir'])
assert predictor.hidden_size == 96
assert predictor.checkpoint['prepared_manifest_sha256'] == plan['training_manifest_sha256']
assert sha256(root / plan['checkpoint']) == plan['checkpoint_sha256']
assert rtl['result'] == dict(matched_steps=10000, cycles=124046,
    reset_replay=True, simulation_seconds=rtl['result']['simulation_seconds'])
names = set(build['sources']) | {
    'model/broad_integer.py', 'model/broad_export_integer.py',
    'model/broad_evaluate_integer.py', 'host/broad_protocol.py',
    'host/verify_broad.py', 'host/ftdi.py', 'host/verify_probe.py',
    'tests/integration/test_broad_integer.py',
    'tests/integration/test_model_integer.py',
    'tests/integration/test_broad_train.py', 'Makefile',
    'experiments/broad-h96-final-study-v1.json',
    'artifacts/broad-h96-final-v1/run-study.py',
    'artifacts/broad-h96-final-v1/run-physical.py',
    'artifacts/broad-h96-final-v1/freeze-selection.py',
}
project_sources = {name: sha256(root / name) for name in sorted(names)}
for name, value in build['sources'].items():
    assert project_sources[name] == value, name
assert rtl['sources']['rtl/broad_core.sv'] == project_sources['rtl/broad_core.sv']
assert build['export']['manifest_sha256'] == sha256(root / plan['export_dir'] / 'manifest.json')
sources, versions = current_identities()
selection = dict(
    frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    plan_sha256=sha256(plan_path),
    recipe_sha256=project_sources['artifacts/broad-h96-final-v1/run-study.py'],
    checkpoint_sha256=plan['checkpoint_sha256'],
    integer_export_manifest_sha256=build['export']['manifest_sha256'],
    build_report=str(build_path.relative_to(root)), build_report_sha256=sha256(build_path),
    bitstream=str(bitstream.relative_to(root)), bitstream_sha256=sha256(bitstream),
    rtl_report=str(rtl_path.relative_to(root)), rtl_report_sha256=sha256(rtl_path),
    model_sources=sources, versions=versions, project_sources=project_sources,
    selection_basis='Best validation model; fixed numeric gate and strict routed50MHz build; new test trials unopened',
    physical_scope=plan['physical_scope'],
)
assert not (root / 'build/broad-h96-final-v1').exists()
assert not (root / 'build/datasets/broad-h96-final-v1').exists()
with (root / 'artifacts/broad-h96-final-v1/selection.json').open('x') as stream:
    stream.write(json.dumps(selection, indent=2, allow_nan=False) + '\n')
print(json.dumps(selection, indent=2))
