"""Root-only SRAM baud experiment, with mandatory original H96 restoration."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from host.ftdi import Ftdi
from host.protocol import build_request
from host.orientation_demo import exchange
from model.broad_integer import GoldenBroadPredictor

ROOT = Path(__file__).resolve().parents[1]
SERIAL = '2023112818'
PROBE = ROOT / 'build/orientation-study-v1/physical-baud-probe-1m'
ORIGINAL = ROOT / 'artifacts/broad-h96-deployment-v2/build/broad.fs'
OUT = ROOT / 'build/orientation-baud-probe-1m-physical.json'
RESTORE_SHA = '3564b2c2ac90d1a38b2d505df4df7945ac665dc17db042ba477274322e463988'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, label):
    command = [str(ROOT / '.tools/tang-toolchain/oss-cad-suite/bin/openFPGALoader'),
               '--board', 'tangprimer25k', '--usb-serial-num', SERIAL,
               '--write-sram', str(path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    (ROOT / f'build/orientation-baud-{label}-load.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{label} SRAM load failed: {result.returncode}')
    return {'argv': command, 'exit_code': result.returncode, 'bitstream_sha256': digest(path)}


def framed(payload):
    checksum = 0
    for value in payload:
        checksum ^= value
    return payload + bytes([checksum])


if OUT.exists():
    raise RuntimeError('Physical evidence already exists')
if digest(ORIGINAL) != RESTORE_SHA:
    raise RuntimeError('Original restoration image identity mismatch')
build = json.loads((PROBE / 'report.json').read_text())
assert build['clock_mhz'] == 50 and build['uart_divisor'] == 50
assert build['timing']['achieved_mhz'] >= 50
assert digest(PROBE / 'probe.fs') == build['bitstream']['sha256']
for name, expected in build['sources'].items():
    assert digest(ROOT / name) == expected, name
report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'source_commit': 'f935482', 'driver_sha256': digest(Path(__file__)), 'usb_serial': SERIAL,
          'probe_build_report_sha256': digest(PROBE / 'report.json'),
          'usb_requested_baudrate': 4000000, 'fpga_nominal_uart_baudrate': 1000000,
          'fpga_clock_mhz': 50, 'uart_divisor': 50, 'checksum': 'XOR8',
          'count_requested': 1000, 'count_verified': 0,
          'flash_written': False, 'higher_H96_sampling_rate_verified': False}
failure = None
try:
    report['probe_load'] = load(PROBE / 'probe.fs', 'probe-1m')
    timings = []
    request_hash, response_hash = hashlib.sha256(), hashlib.sha256()
    started = time.perf_counter()
    with Ftdi(SERIAL, requested_baudrate=4000000) as connection:
        for seq in range(1000):
            a = ((seq * 4051 + 17389) & 65535) - 32768
            b = ((seq * 25173 + 13849) & 65535) - 32768
            address = seq % 256
            if seq == 0:
                a, b, address = -32768, 32767, 0
            elif seq == 1:
                a, b, address = 32767, -32768, 255
            elif seq == 2:
                a, b, address = 0, 32767, 1
            request = build_request(seq, a, b, address)
            assert request == framed(struct.pack('<2sHhhB', b'\xa5\x5a', seq, a, b, address))
            expected = a * b + ((address * 4051 + 17389) & 65535) - 32768
            expected_frame = framed(struct.pack('<2sHi', b'\x5a\xa5', seq, expected))
            step_start = time.perf_counter()
            if connection.write(request) != len(request):
                raise ValueError(f'Incomplete probe request {seq}')
            actual = connection.read(9)
            if actual != expected_frame:
                raise ValueError(f'Probe response mismatch {seq}: {actual.hex()} != {expected_frame.hex()}')
            timings.append((time.perf_counter() - step_start) * 1000)
            request_hash.update(request)
            response_hash.update(actual)
            report['count_verified'] += 1
    elapsed = time.perf_counter() - started
    report.update(elapsed_s=elapsed, diagnostic_verified_transactions_per_s=1000 / elapsed,
                  request_stream_sha256=request_hash.hexdigest(), response_stream_sha256=response_hash.hexdigest(),
                  roundtrip_ms={name: float(value) for name, value in {
                      'mean': np.mean(timings), 'p50': np.percentile(timings, 50),
                      'p95': np.percentile(timings, 95), 'max': max(timings)}.items()})
except Exception as error:
    failure = error
    report['probe_error'] = str(error)
finally:
    try:
        report['restore_load'] = load(ORIGINAL, 'restore-1m')
        predictor = GoldenBroadPredictor.from_export(ROOT / 'artifacts/broad-integer-v1/export')
        assert predictor.hidden_size == 96
        with Ftdi(SERIAL) as connection:
            hidden, raw, cycles = exchange(connection, 65535, [0] * 9, command=2)
            assert (hidden, raw, cycles) == ((0,) * 96, (0,) * 4, 0)
            for index in range(16):
                values = [(((index + 1) * 4051 + (lane + 1) * 13849) & 65535) - 32768
                          for lane in range(9)]
                expected_hidden, expected_raw = predictor.step(values)
                hidden, raw, cycles = exchange(connection, index, values)
                assert hidden == tuple(expected_hidden) and raw == tuple(expected_raw)
                assert cycles == 124046
        report['original_H96_restored'] = True
        report['restore_verified_steps'] = 16
        report['restore_hidden_states_per_step'] = 96
        report['restore_raw_outputs_per_step'] = 4
    except Exception as error:
        report['original_H96_restored'] = False
        report['restore_error'] = str(error)
        if failure is None:
            failure = error
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
if failure:
    raise SystemExit(1)
