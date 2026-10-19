"""UART pin-level recovery and successive-request behaviors."""

# Behaviors covered by P2:
# 1. Successive valid signed boundary requests return corresponding results.
# 2. A bad checksum produces no response and the following valid frame works.
# 3. Active-high reset discards a partial request before a fresh valid frame.
# 4. A leading garbage byte is ignored and exactly one valid response follows.

from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_probe_uart import ROOT, executable
from host.protocol import build_request, parse_response


class ProbeUartRecoveryTests(unittest.TestCase):
    def exchange(self, mode, payload, requests):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            fixture = workspace / "request.hex"
            fixture.write_text("".join(f"{byte:02x}\n" for byte in payload))
            response = workspace / "response.hex"
            simulation = workspace / "cases.vvp"
            compiled = subprocess.run(
                [executable("iverilog"), "-g2012", "-s", "probe_uart_cases_tb",
                 "-o", str(simulation), "rtl/uart.sv", "rtl/probe_top.sv",
                 "tests/integration/probe_uart_cases_tb.sv"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            simulated = subprocess.run(
                [executable("vvp"), str(simulation), f"+REQ={fixture}",
                 f"+RESP={response}", f"+MODE={mode}", f"+N={len(payload)}",
                 f"+FRAMES={len(requests)}"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(simulated.returncode, 0, simulated.stdout + simulated.stderr)
            captured = bytes.fromhex(response.read_text())
            actual = [parse_response(captured[index:index + 9])
                      for index in range(0, len(captured), 9)]
            expected = [
                (seq, a * b + ((address * 4051 + 17389) & 65535) - 32768)
                for seq, a, b, address in requests
            ]
            self.assertEqual(actual, expected)

    def test_successive_requests_preserve_signed_boundary_results_and_sequences(self):
        requests = [(1, -32768, 32767, 0), (65535, 32767, -32768, 255),
                    (234, 0, 32767, 17)]
        self.exchange(0, b"".join(build_request(*item) for item in requests), requests)

    def test_bad_checksum_is_silent_and_following_valid_request_succeeds(self):
        request = (725, -123, 17, 17)
        valid = build_request(*request)
        corrupt = valid[:-1] + bytes([valid[-1] ^ 1])
        self.exchange(1, corrupt + valid, [request])

    def test_reset_discards_partial_request_and_accepts_fresh_request(self):
        request = (826, -7, -11, 255)
        partial = build_request(111, 123, 456, 0)[:5]
        self.exchange(2, partial + build_request(*request), [request])

    def test_leading_garbage_byte_is_ignored_before_valid_request(self):
        request = (927, 23, -19, 0)
        self.exchange(3, b"\x00" + build_request(*request), [request])
