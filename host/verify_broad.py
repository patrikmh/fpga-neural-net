"""Verify a bound BROAD integer vector package on one explicitly named USB device."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

from host.broad_protocol import build_request, parse_response
from host.ftdi import DEFAULT_LIBRARY, Ftdi
from host.verify_probe import usb_serial_number
from model.broad_integer import GoldenBroadPredictor, runtime_versions, sha256


ROOT = Path(__file__).resolve().parents[1]


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return result


def snapshot(export_dir, vectors_dir, library):
    hashes = {}
    for label, directory in (('export', export_dir), ('vectors', vectors_dir)):
        for path in directory.rglob('*'):
            if path.is_file():
                hashes[f'{label}/{path.relative_to(directory)}'] = sha256(path)
    for name in ('host/verify_broad.py', 'host/broad_protocol.py', 'host/ftdi.py',
                 'host/verify_probe.py', 'model/broad_integer.py', 'rtl/broad_core.sv',
                 'rtl/broad_top.sv', 'rtl/uart.sv'):
        hashes[name] = sha256(ROOT / name)
    hashes['libftdi'] = sha256(library)
    return hashes


def summary(values):
    ordered = sorted(values)
    return {'min': ordered[0], 'mean': statistics.mean(values),
            'p95': ordered[math.ceil(len(values) * .95) - 1], 'max': ordered[-1]}


def exchange(connection, seq, inputs, hidden_size, command=1):
    request = build_request(seq, inputs, command)
    if connection.write(request) != len(request):
        raise ValueError(f'Incomplete BROAD request write at sequence {seq}')
    response = connection.read(18 + 2 * hidden_size)
    actual_seq, hidden, raw, cycles = parse_response(response, hidden_size)
    if actual_seq != seq:
        raise ValueError(f'BROAD sequence mismatch: expected {seq}, received {actual_seq}')
    return hidden, raw, cycles


def verify(args):
    if args.output is not None:
        output = args.output.resolve()
        for directory in (args.export_dir.resolve(), args.vectors_dir.resolve()):
            if output == directory or directory in output.parents:
                raise ValueError('Output must be outside frozen export and vector directories')
        if output.exists():
            raise ValueError(f'Output already exists: {output}')
    predictor = GoldenBroadPredictor.from_export(args.export_dir)
    artifact = predictor.manifest
    size = predictor.hidden_size
    manifest_path = args.vectors_dir / 'manifest.json'
    vectors = json.loads(manifest_path.read_text())
    if vectors['export_manifest_sha256'] != sha256(args.export_dir / 'manifest.json'):
        raise ValueError('Vector/export manifest SHA256 identity mismatch')
    if vectors['checkpoint_sha256'] != artifact['checkpoint_sha256']:
        raise ValueError('Vector/checkpoint SHA256 identity mismatch')
    if vectors['vectors_sha256'] != sha256(args.vectors_dir / 'vectors.hex'):
        raise ValueError('Vectors SHA256 identity mismatch')
    if (vectors['hidden_size'] != size or vectors['initial_hidden'] != [0] * size
            or vectors['numeric'] != artifact['numeric']):
        raise ValueError('Vector architecture, initial state or numeric format mismatch')
    rows = (args.vectors_dir / 'vectors.hex').read_text().splitlines()
    if type(vectors['count']) is not int or len(rows) != vectors['count'] or args.count > len(rows):
        raise ValueError('Vector count mismatch or requested count exceeds available vectors')
    decoded = []
    width = 4 * (9 + size + 4)
    for row in rows[:args.count]:
        if len(row) != width or any(char not in '0123456789abcdefABCDEF' for char in row):
            raise ValueError('Invalid signed-16 BROAD vector row')
        words = [int(row[offset:offset + 4], 16) for offset in range(0, width, 4)]
        decoded.append([word - (65536 if word >= 32768 else 0) for word in words])
    library = Path(args.libftdi) if args.libftdi else DEFAULT_LIBRARY
    before = snapshot(args.export_dir, args.vectors_dir, library)
    roundtrips, compute = [], []
    started = time.perf_counter()
    with Ftdi(args.usb_serial, str(library), requested_baudrate=args.usb_requested_baudrate) as connection:
        hidden, raw, cycles = exchange(connection, 65535, [0] * 9, size, command=2)
        if (hidden, raw, cycles) != ((0,) * size, (0,) * 4, 0):
            raise ValueError('BROAD RESET acknowledgement did not clear state, output and cycles')
        for index, row in enumerate(decoded):
            expected_hidden, expected_raw = predictor.step(row[:9])
            if expected_hidden + expected_raw != row[9:]:
                raise ValueError(f'Stored vector differs from golden model at index {index}')
            step_started = time.perf_counter()
            hidden, raw, cycles = exchange(connection, index & 65535, row[:9], size)
            elapsed = (time.perf_counter() - step_started) * 1000
            if hidden != tuple(expected_hidden) or raw != tuple(expected_raw):
                raise ValueError(f'BROAD state/output mismatch at index {index}')
            if cycles != args.expected_cycles:
                raise ValueError(f'BROAD cycle count mismatch at index {index}: {cycles}')
            roundtrips.append(elapsed)
            compute.append(cycles / 50)
    if snapshot(args.export_dir, args.vectors_dir, library) != before:
        raise ValueError('Code or artifact SHA256 identity changed during verification')
    return {'count_requested': args.count, 'count_verified': len(roundtrips), 'hidden_size': size,
            'reset_verified': True, 'expected_cycles': args.expected_cycles, 'transport': 'libftdi',
            'usb_serial': args.usb_serial, 'usb_requested_baudrate': args.usb_requested_baudrate,
            'uart_divisor': args.uart_divisor, 'expected_fpga_baudrate': 50000000 / args.uart_divisor,
            'baudrate': 115200 if args.uart_divisor == 434 else 50000000 / args.uart_divisor,
            'clock_mhz': 50, 'elapsed_s': time.perf_counter() - started,
            'compute_us': summary(compute), 'roundtrip_ms': summary(roundtrips),
            'export_manifest_sha256': before['export/manifest.json'],
            'vectors_manifest_sha256': before['vectors/manifest.json'],
            'checkpoint_sha256': artifact['checkpoint_sha256'],
            'versions': runtime_versions(), 'provenance': before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usb-serial', required=True, type=usb_serial_number)
    parser.add_argument('--export-dir', required=True, type=Path)
    parser.add_argument('--vectors-dir', required=True, type=Path)
    parser.add_argument('--count', type=positive_integer, default=10000)
    parser.add_argument('--expected-cycles', type=positive_integer, required=True)
    parser.add_argument('--libftdi')
    parser.add_argument('--usb-requested-baudrate', type=positive_integer, default=460800)
    parser.add_argument('--uart-divisor', type=positive_integer, default=434)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = verify(args)
        encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                stream.write(encoded)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'BROAD verification failed: {error}', file=sys.stderr)
        return 1
    print(encoded, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
