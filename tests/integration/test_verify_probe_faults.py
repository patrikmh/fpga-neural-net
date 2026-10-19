"""Fault-injection regressions at the verifier's external serial boundary."""

# Behaviors covered:
# 1. Corrupt reply checksums cannot be reported as successful verification.
# 2. Wrong reply sequences cannot be reported as successful verification.
# 3. Wrong arithmetic results cannot be reported as successful verification.
# 4. Partial or absent replies time out without a successful report.
# 5. One valid reply produces finite statistics for a single observation.

import json
import math
import os
from pathlib import Path
import queue
import select
import struct
import subprocess
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]


def serial_peer(master, stop, outcomes, fault):
    try:
        request = bytearray()
        deadline = time.monotonic() + 5
        while len(request) < 10:
            if stop.is_set():
                raise AssertionError("Verifier exited before sending its request")
            if time.monotonic() >= deadline:
                raise AssertionError("Verifier did not send its request within five seconds")
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                request.extend(os.read(master, 1024))
        checksum = 0
        for byte in request:
            checksum ^= byte
        if len(request) != 10 or request[:2] != b"\xa5\x5a" or checksum:
            raise AssertionError(f"Invalid outgoing request: {request.hex()}")
        sequence, a, b, address = struct.unpack("<HhhB", request[2:9])
        if (sequence, a, b, address) != (0, -32768, 32767, 0):
            raise AssertionError("Verifier sent an unexpected initial vector")
        result = a * b + ((address * 4051 + 17389) & 65535) - 32768
        if fault == "sequence":
            sequence += 1
        if fault == "result":
            result += 1
        response = b"\x5a\xa5" + struct.pack("<Hi", sequence, result)
        checksum = 0
        for byte in response:
            checksum ^= byte
        response += bytes([checksum ^ (fault == "checksum")])
        if fault == "partial":
            response = response[:4]
        if fault == "absent":
            response = b""
        while response:
            response = response[os.write(master, response):]
        outcomes.put(None)
    except Exception as error:
        outcomes.put(error)


class VerifyProbeFaultTests(unittest.TestCase):
    def run_exchange(self, fault):
        master, slave = os.openpty()
        stop = threading.Event()
        outcomes = queue.Queue()
        peer = threading.Thread(target=serial_peer, args=(master, stop, outcomes, fault))
        try:
            peer.start()
            try:
                result = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), "-m", "host.verify_probe",
                     "--port", os.ttyname(slave), "--count", "1"],
                    cwd=ROOT, capture_output=True, text=True, timeout=8,
                )
            finally:
                stop.set()
                peer.join(timeout=2)
            self.assertFalse(peer.is_alive(), "External serial peer must terminate")
            self.assertFalse(outcomes.empty(), "Peer must process the actual request")
            error = outcomes.get_nowait()
            self.assertIsNone(error, f"External serial peer failed: {error}")
            return result
        finally:
            os.close(master)
            os.close(slave)

    def assert_rejected(self, fault, diagnostic):
        result = self.run_exchange(fault)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(diagnostic, result.stderr.lower())
        if result.stdout.strip():
            self.assertNotIn("count_verified", json.loads(result.stdout))

    def test_bad_response_checksum_is_rejected(self):
        self.assert_rejected("checksum", "checksum")

    def test_valid_checksum_with_wrong_sequence_is_rejected(self):
        self.assert_rejected("sequence", "sequence mismatch")

    def test_valid_checksum_with_wrong_result_is_rejected(self):
        self.assert_rejected("result", "result mismatch")

    def test_partial_or_absent_response_times_out_without_success(self):
        for fault in ("partial", "absent"):
            with self.subTest(reply=fault):
                self.assert_rejected(fault, "timeout")

    def test_one_verified_response_has_finite_single_observation_statistics(self):
        result = self.run_exchange("valid")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["count_requested"], 1)
        self.assertEqual(report["count_verified"], 1)
        values = [report["roundtrip_ms"][name] for name in ("min", "mean", "p95", "max")]
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))
        self.assertEqual(values, [values[0]] * 4)


if __name__ == "__main__":
    unittest.main()
