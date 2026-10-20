"""THICKENING: 2 Mbaud probe UART pins and a divisor-bound 50 MHz build.

Behaviors covered:
1. Divisor25 preserves the host sequence, signed multiply+ROM/RAM result and XOR frame.
2. A real routed probe binds divisor25 to synthesis/report/bitstream; invalid values fail early.
"""

from functools import reduce
import hashlib
import json
import operator
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from integration.test_probe_uart import executable


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv/bin/python')
TOOLS = ROOT / '.tools/tang-toolchain/oss-cad-suite/bin'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProbeFastUartTests(unittest.TestCase):
    def test_divisor25_uart_pins_preserve_signed_math_rom_ram_sequence_and_xor_reply(self):
        from host.protocol import build_request, parse_response

        sequence, a, b, address = 0xfea5, -32768, 32767, 231
        payload = struct.pack('<2sHhhB', b'\xa5\x5a', sequence, a, b, address)
        expected_request = payload + bytes([reduce(operator.xor, payload)])
        request = build_request(sequence, a, b, address)
        self.assertEqual(request, expected_request)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            bench = workspace / 'probe_fast_tb.sv'
            source = (ROOT / 'tests/integration/probe_uart_tb.sv').read_text()
            source = source.replace('BIT_NS = 1.0e9 / 115200.0', 'BIT_NS = 500.0')
            source = source.replace('probe_top dut(', 'probe_top #(.UART_DIVISOR(25)) dut(')
            bench.write_text(source)
            simulation = workspace / 'probe.vvp'
            compiled = subprocess.run([executable('iverilog'), '-g2012', '-s', 'probe_uart_tb',
                '-o', str(simulation), 'rtl/uart.sv', 'rtl/probe_top.sv', str(bench)],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, '2 Mbaud probe configuration must elaborate:\n'
                             + compiled.stdout + compiled.stderr)
            requests, responses = workspace / 'request.hex', workspace / 'response.hex'
            requests.write_text(''.join(f'{byte:02x}\n' for byte in request))
            result = subprocess.run([executable('vvp'), str(simulation), f'+REQ={requests}', f'+RESP={responses}'],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, 'Actual 2 Mbaud UART pin exchange must complete:\n'
                             + result.stdout + result.stderr)
            response = bytes.fromhex(responses.read_text())
            rom = ((address*4051 + 17389) & 65535) - 32768
            response_payload = struct.pack('<2sHi', b'\x5a\xa5', sequence, a*b + rom)
            expected_response = response_payload + bytes([reduce(operator.xor, response_payload)])
            self.assertEqual(response, expected_response)
            self.assertEqual(parse_response(response), (sequence, a*b + rom))

    def test_divisor25_probe_build_binds_routed_50mhz_source_memory_and_bitstream_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for invalid in ('0', '-1', '1.5'):
                with self.subTest(invalid_divisor=invalid):
                    rejected = subprocess.run([PYTHON, 'scripts/build.py', '--build-dir',
                        str(workspace / ('invalid-'+invalid)), '--uart-divisor', invalid],
                        cwd=ROOT, env={**os.environ, 'PATH': ''}, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn('--uart-divisor', rejected.stderr)
                    self.assertNotIn('Required tool is missing', rejected.stderr)
            output = workspace / 'build'
            result = subprocess.run([PYTHON, 'scripts/build.py', '--build-dir', str(output), '--uart-divisor', '25'],
                cwd=ROOT, env={**os.environ, 'PATH': str(TOOLS)+os.pathsep+os.environ.get('PATH', '')},
                capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, 'Actual divisor25 probe build must complete:\n'
                             + result.stdout + result.stderr)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual((report['top'], report['clock_mhz'], report['uart_divisor']), ('probe_top', 50, 25))
            self.assertEqual(report['expected_fpga_baudrate'], 2000000)
            self.assertEqual(report['timing']['constrained_mhz'], 50)
            self.assertGreaterEqual(report['timing']['achieved_mhz'], 50)
            synthesized = json.loads((output / 'synth.json').read_text())
            self.assertEqual(int(synthesized['modules']['probe_top']['parameter_default_values']['UART_DIVISOR'], 2), 25)
            for name in ('rtl/uart.sv', 'rtl/probe_top.sv', 'rtl/probe_rom.hex',
                         'constraints/primer25k.cst', 'scripts/build.py'):
                self.assertEqual(report['sources'][name], sha256(ROOT / name))
            rom, ram = report['memories']['rom'], report['memories']['ram']
            self.assertGreater(rom['block_count'], 0)
            self.assertGreater(ram['block_count'], 0)
            self.assertTrue(set(rom['origins']).isdisjoint(ram['origins']))
            bitstream = (output / report['bitstream']['file']).resolve()
            self.assertEqual(bitstream.parent, output.resolve())
            self.assertGreater(bitstream.stat().st_size, 0)
            self.assertEqual(report['bitstream']['sha256'], sha256(bitstream))


if __name__ == '__main__':
    unittest.main()
