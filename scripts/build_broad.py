"""Build a checked BROAD export with banked ROM and routed 50 MHz evidence."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.broad_integer import GoldenBroadPredictor
from scripts.build import DEVICE, sha256, timing_evidence


SOURCES = ('rtl/uart.sv', 'rtl/broad_core.sv', 'rtl/broad_top.sv',
           'constraints/primer25k.cst', 'scripts/build_broad.py', 'scripts/build.py',
           'model/broad_integer.py')


def positive_finite(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Expected a finite positive number')
    return number


def heap_beta(value):
    number = positive_finite(value)
    if number > 1:
        raise argparse.ArgumentTypeError('Heap beta must be in (0, 1]')
    return number


def run_stage(name, command, output, timeout):
    log = output / f'{name}.log'
    print(f'{name}: running (log: {log})', flush=True)
    with log.open('w') as stream:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'{name} exceeded {timeout}s; see {log}') from error
    if completed.returncode:
        raise RuntimeError(f'{name} failed with status {completed.returncode}; see {log}')
    print(f'{name}: completed', flush=True)


def file_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in directory.rglob('*') if path.is_file()}


def memory_evidence(routed, bank_count):
    cells = routed['modules']['top']['cells']
    patterns = {'weights': r'core\.weight_banks\[\d+\]\.memory\.\d+\.\d+(?:_AUX)?',
                'tanh': r'core\.tanh_rom\.\d+\.\d+(?:_AUX)?',
                'sigmoid': r'core\.sigmoid_rom\.\d+\.\d+(?:_AUX)?'}
    memories, origins = {}, set()
    for name, expected in (('weights', bank_count), ('tanh', 4), ('sigmoid', 3)):
        matches = {origin: cell for origin, cell in cells.items()
                   if re.fullmatch(patterns[name], origin) and cell['type'] in ('SP', 'SPX9')
                   and cell.get('attributes', {}).get('NEXTPNR_BEL', '').endswith('/BSRAM')}
        if len(matches) != expected or origins.intersection(matches):
            raise RuntimeError(f'Unrecognized routed {name} memory mapping: {sorted(matches)}')
        origins.update(matches)
        memories[name] = {'block_count': len(matches), 'origins': sorted(matches),
                          'cell_types': sorted({cell['type'] for cell in matches.values()})}
    return memories


def build(args):
    export_dir, output = args.export_dir.resolve(), args.build_dir.resolve()
    if output == export_dir or export_dir in output.parents:
        raise ValueError('Build output must be outside the frozen export')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Build output directory must be empty: {output}')
    predictor = GoldenBroadPredictor.from_export(export_dir)
    size = predictor.hidden_size
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    export_hashes = file_hashes(export_dir)
    tools, identities = {}, {}
    for name in ('yosys', 'nextpnr-himbaechel', 'gowin_pack'):
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f'Required tool is missing: {name}')
        tools[name] = path
        if name == 'gowin_pack':
            identities[name] = f'{path} sha256:{sha256(Path(path))}'
        else:
            result = subprocess.run([path, '-V' if name == 'yosys' else '--version'],
                                    capture_output=True, text=True, check=True, timeout=10)
            identities[name] = (result.stdout + result.stderr).strip()
            if not identities[name]:
                raise RuntimeError(f'Tool identity is empty: {name}')
    tool_hashes = {name: sha256(Path(path)) for name, path in tools.items()}
    output.mkdir(parents=True, exist_ok=True)
    rom_dir = output / 'rom'
    rom_dir.mkdir()
    words = (export_dir / 'weights.hex').read_text().split()
    banks = []
    for number, offset in enumerate(range(0, len(words), 1024)):
        valid = words[offset:offset + 1024]
        path = rom_dir / f'weights_{number:03d}.hex'
        path.write_text('\n'.join(valid + ['0000'] * (1024 - len(valid))) + '\n')
        banks.append({'file': str(path.relative_to(output)), 'sha256': sha256(path), 'valid_words': len(valid)})
    staged_hashes = file_hashes(rom_dir)
    synth, routed = output / 'synth.json', output / 'routed.json'
    timing_path, bitstream = output / 'timing.json', output / 'broad.fs'
    parameters = (f'-set HIDDEN_SIZE {size} -set WEIGHTS_DIR {json.dumps(str(rom_dir))} '
                  f'-set TANH_FILE {json.dumps(str(export_dir / "tanh.hex"))} '
                  f'-set SIGMOID_FILE {json.dumps(str(export_dir / "sigmoid.hex"))}')
    run_stage('yosys', [tools['yosys'], '-p',
        'read_verilog -sv rtl/uart.sv rtl/broad_core.sv rtl/broad_top.sv; '
        f'chparam {parameters} broad_top; '
        'synth_gowin -top broad_top -family gw5a -nolutram; '
        f'write_json {json.dumps(str(synth))}'], output, args.stage_timeout)
    command = [tools['nextpnr-himbaechel'], '--json', str(synth), '--device', DEVICE,
        '--vopt', 'family=GW5A-25A', '--vopt', f'cst={ROOT / SOURCES[3]}', '--freq', '50',
        '--placer', args.placer, '--router', 'router2', '--seed', str(args.seed),
        '--report', str(timing_path), '--write', str(routed)]
    if args.placer == 'heap':
        command.extend(['--placer-heap-beta', str(args.heap_beta)])
    run_stage('nextpnr', command, output, args.stage_timeout)
    memories = memory_evidence(json.loads(routed.read_text()), len(banks))
    raw_timing = json.loads(timing_path.read_text())
    timing = timing_evidence(raw_timing)
    if sum(item['block_count'] for item in memories.values()) != raw_timing['utilization']['BSRAM']['used']:
        raise RuntimeError('Routed ROM origins disagree with BSRAM utilization')
    run_stage('gowin_pack', [tools['gowin_pack'], '-d', 'GW5A-25A', '-o', str(bitstream), str(routed)],
              output, args.stage_timeout)
    if not bitstream.is_file() or not bitstream.stat().st_size:
        raise RuntimeError('Packer did not produce a nonempty bitstream')
    if source_hashes != {name: sha256(ROOT / name) for name in SOURCES}:
        raise RuntimeError('Build sources changed during build')
    if export_hashes != file_hashes(export_dir) or staged_hashes != file_hashes(rom_dir):
        raise RuntimeError('Frozen export or staged ROM banks changed during build')
    if tool_hashes != {name: sha256(Path(path)) for name, path in tools.items()}:
        raise RuntimeError('Tool executable changed during build')
    report = {'top': 'broad_top', 'device': DEVICE, 'hidden_size': size, 'clock_mhz': 50, 'router': 'router2',
        'placement': {'placer': args.placer, 'seed': args.seed, 'heap_beta': args.heap_beta},
        'stage_timeout_s': args.stage_timeout, 'timing': timing, 'memories': memories,
        'utilization': raw_timing['utilization'], 'sources': source_hashes, 'banks': banks,
        'tools': identities, 'tool_sha256': tool_hashes,
        'export': {'manifest_sha256': export_hashes['manifest.json'],
                   'files': {name: export_hashes[name] for name in predictor.manifest['files']}},
        'bitstream': {'file': bitstream.name, 'sha256': sha256(bitstream)}}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Verified BROAD H{size}: {timing["achieved_mhz"]:.2f} MHz; report: {output / "report.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, default=ROOT / 'artifacts/broad-integer-v1/export')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/broad')
    parser.add_argument('--placer', choices=('heap', 'sa'), default='heap')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--heap-beta', type=heap_beta, default=.9)
    parser.add_argument('--stage-timeout', type=positive_finite, default=600)
    args = parser.parse_args()
    try:
        build(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'BROAD build failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
