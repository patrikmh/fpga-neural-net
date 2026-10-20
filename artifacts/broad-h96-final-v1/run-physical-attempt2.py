"""SRAM-load the frozen H96 bitstream and verify four independent USB prefixes."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
sys.path.insert(0, str(root))
from model.broad_integer import sha256

selection_path = root / 'artifacts/broad-h96-final-v1/selection.json'
selection = json.loads(selection_path.read_text())
selection_hash = sha256(selection_path)
study = root / 'build/broad-h96-final-v1'
cpu_report = json.loads((study / 'report.json').read_text())
assert cpu_report['selection_sha256'] == selection_hash
assert cpu_report['test_used_for_selection'] is False
plan = json.loads((root / 'experiments/broad-h96-final-study-v1.json').read_text())
count = plan['physical_prefix_steps_per_trial']
assert count == 1000 and plan['physical_trials'] == [4, 8, 12, 31]
rtl = json.loads((root / selection['rtl_report']).read_text())
cycles = rtl['result']['cycles']
bitstream = root / selection['bitstream']
loader = root / '.tools/tang-toolchain/oss-cad-suite/bin/openFPGALoader'
loader_binary = loader.parent.parent / 'libexec/openFPGALoader'


def verify_frozen():
    assert sha256(selection_path) == selection_hash
    assert sha256(bitstream) == selection['bitstream_sha256']
    assert sha256(root / selection['build_report']) == selection['build_report_sha256']
    assert sha256(root / selection['rtl_report']) == selection['rtl_report_sha256']
    assert sha256(root / 'experiments/broad-h96-final-study-v1.json') == selection['plan_sha256']
    for name, value in selection['project_sources'].items():
        assert sha256(root / name) == value, name


verify_frozen()
output = root / 'build/broad-h96-physical-v2'
output.mkdir(exist_ok=False)
command = [str(loader), '-b', 'tangprimer25k', '--usb-serial-num', '2023112818',
           '--write-sram', str(bitstream)]
preflight = dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selection_sha256=selection_hash, cpu_report_sha256=sha256(study / 'report.json'),
    recipe_sha256=sha256(Path(__file__)), loader_sha256=sha256(loader),
    loader_binary_sha256=sha256(loader_binary),
    loader_version=subprocess.check_output([str(loader), '--version'], text=True).strip(),
    bitstream_sha256=sha256(bitstream), programming_command=command,
    programming_scope='SRAM only', usb_serial='2023112818',
    previous_attempt='build/broad-h96-physical-v1',
    previous_attempt_result='Loader failed before programming: unable to fetch serial number',
    recovery='Subsequent serial-qualified JTAG detection passed; cause of recovery unknown',
    selected_recipe_sha256=selection['project_sources']['artifacts/broad-h96-final-v1/run-physical.py'])
(output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
print('Loading frozen H96 bitstream to SRAM on serial2023112818', flush=True)
with (output / 'program.log').open('x') as log:
    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
results = []
for trial_id in plan['physical_trials']:
    vectors = study / 'usb-vectors' / f'{trial_id:02d}'
    manifest = json.loads((vectors / 'manifest.json').read_text())
    assert manifest['trial_id'] == trial_id and manifest['count'] == count
    assert manifest['rtl_report_sha256'] == selection['rtl_report_sha256']
    assert manifest['test_spec_sha256'] == cpu_report['test_spec_sha256']
    report_path = output / f'usb-{trial_id:02d}.json'
    command = ['make', 'broad-verify', 'USB_SERIAL=2023112818',
        f'BROAD_EXPORT_DIR={root / plan["export_dir"]}', f'BROAD_VECTORS_DIR={vectors}',
        f'BROAD_COUNT={count}', f'BROAD_EXPECTED_CYCLES={cycles}', f'BROAD_OUTPUT={report_path}']
    print(f'Verifying RESET+{count} continuous steps for trial{trial_id:02d}', flush=True)
    with (output / f'usb-{trial_id:02d}.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    report = json.loads(report_path.read_text())
    assert report['count_verified'] == count and report['reset_verified']
    assert report['hidden_size'] == 96 and report['expected_cycles'] == cycles
    assert report['vectors_manifest_sha256'] == sha256(vectors / 'manifest.json')
    assert report['export_manifest_sha256'] == selection['integer_export_manifest_sha256']
    results.append(dict(trial_id=trial_id, report=report_path.name,
        report_sha256=sha256(report_path), count_verified=report['count_verified'],
        compute_us=report['compute_us'], roundtrip_ms=report['roundtrip_ms']))
    verify_frozen()
assert sha256(Path(__file__)) == preflight['recipe_sha256']
assert sha256(loader) == preflight['loader_sha256']
assert sha256(loader_binary) == preflight['loader_binary_sha256']
assert sha256(study / 'report.json') == preflight['cpu_report_sha256']
result = dict(preflight, completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    trials=results, total_verified_steps=sum(entry['count_verified'] for entry in results),
    scope=plan['physical_scope'], physical_verification=True,
    remaining_stability_gate='Prior scalar600sUSB endurance gate remains open; this prefix proof does not close it')
(output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
