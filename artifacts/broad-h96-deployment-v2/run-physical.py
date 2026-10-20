"""Verify a bit-exact implementation repair against the unchanged frozen study."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model.broad_integer import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--rtl-report', type=Path)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    selection_path = ROOT / 'artifacts/broad-h96-final-v1/selection.json'
    selection = json.loads(selection_path.read_text())
    cpu_path = ROOT / 'artifacts/broad-h96-final-v1/report.json'
    cpu = json.loads(cpu_path.read_text())
    report_path = build / 'report.json'
    report = json.loads(report_path.read_text())
    bitstream = build / report['bitstream']['file']
    assert cpu['selection_sha256'] == sha256(selection_path)
    assert cpu['test_used_for_selection'] is False
    assert report['hidden_size'] == 96 and report['clock_mhz'] == 50
    assert report['timing']['achieved_mhz'] >= 50
    assert report['memories']['weights']['block_count'] == 41
    assert report['utilization']['BSRAM']['used'] == 48
    assert report['export']['manifest_sha256'] == selection['integer_export_manifest_sha256']
    assert sha256(bitstream) == report['bitstream']['sha256']
    # The new core is an implementation repair, not another model selection.
    frozen = {ROOT / name: digest for name, digest in selection['project_sources'].items()
              if name != 'rtl/broad_core.sv'}
    for name, digest in report['sources'].items():
        if name != 'rtl/broad_core.sv':
            assert digest == selection['project_sources'][name], name
    frozen.update({ROOT / name: digest for name, digest in report['sources'].items()})
    export_path = ROOT / 'artifacts/broad-integer-v1/export/manifest.json'
    export = json.loads(export_path.read_text())
    assert sha256(export_path) == selection['integer_export_manifest_sha256']
    assert export['checkpoint_sha256'] == report['export']['files']['float_checkpoint.pt']
    frozen[export_path] = sha256(export_path)
    for path, digest in frozen.items():
        assert sha256(path) == digest, str(path)
    for path in (selection_path, cpu_path, report_path, bitstream, Path(__file__)):
        frozen[path] = sha256(path)
    cycles = 124046
    if args.rtl_report:
        rtl_path = args.rtl_report.resolve()
        rtl = json.loads(rtl_path.read_text())
        assert rtl['result']['cycles'] == cycles
        assert rtl['result']['matched_steps'] == 10000 and rtl['result']['reset_replay']
        assert rtl['sources']['rtl/broad_core.sv'] == report['sources']['rtl/broad_core.sv']
        stage_path = rtl_path.parent / 'stage-manifest.json'
        assert sha256(stage_path) == rtl['stage_manifest_sha256']
        assert json.loads(stage_path.read_text())['export_manifest_sha256'] == selection['integer_export_manifest_sha256']
        frozen[stage_path] = sha256(stage_path)
        frozen[rtl_path] = sha256(rtl_path)
    elif not args.quick:
        parser.error('--rtl-report is required for the full proof')
    packages = [('validation-prefix', ROOT / 'build/broad-usb-vectors-v1',
                 16 if args.quick else 10000)]
    if not args.quick:
        packages.append(('validation-motion', ROOT / 'artifacts/broad-validation-motion-v1', 1000))
        packages.extend((f'test-{trial:02d}', ROOT / f'artifacts/broad-h96-final-v1/usb-vectors/{trial:02d}', 1000)
                        for trial in (4, 8, 12, 31))
    for _, package, count in packages:
        manifest = json.loads((package / 'manifest.json').read_text())
        assert manifest['count'] >= count
        assert manifest['export_manifest_sha256'] == selection['integer_export_manifest_sha256']
        assert manifest['vectors_sha256'] == sha256(package / 'vectors.hex')
        for name in ('manifest.json', 'vectors.hex'):
            frozen[package / name] = sha256(package / name)
    loader = ROOT / '.tools/tang-toolchain/oss-cad-suite/bin/openFPGALoader'
    for path in (loader, loader.parent.parent / 'libexec/openFPGALoader'):
        frozen[path] = sha256(path)
    command = [str(loader), '-b', 'tangprimer25k', '--usb-serial-num', '2023112818',
               '--write-sram', str(bitstream)]
    output.mkdir(parents=True, exist_ok=False)
    preflight = dict(started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_selection_sha256=sha256(selection_path), parent_cpu_report_sha256=sha256(cpu_path),
        implementation_repair_only=True, model_reselected=False,
        source_changes_from_selection={'rtl/broad_core.sv': {
            'before': selection['project_sources']['rtl/broad_core.sv'],
            'after': report['sources']['rtl/broad_core.sv']}},
        quick_diagnostic=args.quick, programming_scope='SRAM only', usb_serial='2023112818',
        programming_command=command, frozen_files={str(p.relative_to(ROOT)): h for p, h in frozen.items()},
        loader_version=subprocess.check_output([str(loader), '--version'], text=True).strip(),
        vector_provenance='Original numerical vectors preserved; older RTL fields describe their original packaging, not this deployment',
        scope='RESET plus continuous bit-exact state/raw/cycle verification; movement window resets at index11449; no physical full-trial accuracy claim')
    (output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    with (output / 'program.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    results = []
    for label, package, count in packages:
        path = output / f'{label}.json'
        command = ['make', 'broad-verify', 'USB_SERIAL=2023112818',
            'BROAD_EXPORT_DIR=artifacts/broad-integer-v1/export', f'BROAD_VECTORS_DIR={package}',
            f'BROAD_COUNT={count}', f'BROAD_EXPECTED_CYCLES={cycles}', f'BROAD_OUTPUT={path}']
        print(f'Verifying {label}: RESET + {count} steps', flush=True)
        with (output / f'{label}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        verified = json.loads(path.read_text())
        assert verified['reset_verified'] and verified['count_verified'] == count
        assert verified['expected_cycles'] == cycles and verified['hidden_size'] == 96
        assert verified['provenance']['rtl/broad_core.sv'] == report['sources']['rtl/broad_core.sv']
        results.append(dict(label=label, report=path.name, sha256=sha256(path), count_verified=count))
        for frozen_path, digest in frozen.items():
            assert sha256(frozen_path) == digest, str(frozen_path)
    result = dict(preflight, completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        results=results, total_verified_steps=sum(r['count_verified'] for r in results),
        physical_verification=True,
        remaining_stability_gate='Earlier scalar 600s USB endurance gate remains open')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
