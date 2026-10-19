"""Verification CLI tracer against an independent serial device on a PTY."""

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
VECTORS = ((0, -32768, 32767, 0), (1, 32767, -32768, 255), (2, 0, 32767, 1))


def respond_to_requests(master, stop, errors):
    """Represent only the external serial device, independently of host code."""
    pending = bytearray()
    deadline = time.monotonic() + 5
    try:
        for expected in VECTORS:
            while len(pending) < 10:
                if stop.is_set():
                    return
                if time.monotonic() >= deadline:
                    raise AssertionError("Verification CLI did not send all three requests")
                readable, _, _ = select.select([master], [], [], 0.05)
                if readable:
                    pending.extend(os.read(master, 1024))
            frame = bytes(pending[:10])
            del pending[:10]
            if frame[:2] != b"\xa5\x5a":
                raise AssertionError(f"Unexpected request header: {frame.hex()}")
            checksum = 0
            for byte in frame:
                checksum ^= byte
            if checksum:
                raise AssertionError(f"Invalid request checksum: {frame.hex()}")
            actual = struct.unpack("<HhhB", frame[2:9])
            if actual != expected:
                raise AssertionError(f"Expected request {expected}, received {actual}")
            seq, a, b, address = actual
            rom = ((address * 4051 + 17389) & 65535) - 32768
            response = b"\x5a\xa5" + struct.pack("<Hi", seq, a * b + rom)
            checksum = 0
            for byte in response:
                checksum ^= byte
            payload = response + bytes([checksum])
            while payload:
                payload = payload[os.write(master, payload):]
        errors.put(None)
    except Exception as error:
        errors.put(error)


class VerifyProbeCliTracerTests(unittest.TestCase):
    def test_verification_cli_checks_three_serial_vectors_and_reports_roundtrip_timings(self):
        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        port = os.ttyname(slave)
        stop = threading.Event()
        errors = queue.Queue()
        peer = threading.Thread(target=respond_to_requests, args=(master, stop, errors))
        peer.start()
        try:
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-m", "host.verify_probe",
                 "--port", port, "--count", "3"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
        finally:
            stop.set()
            peer.join(timeout=2)

        self.assertEqual(
            result.returncode, 0,
            "Verification CLI must complete a real serial exchange:\n"
            + result.stdout + result.stderr,
        )
        self.assertFalse(peer.is_alive(), "External serial peer must terminate")
        self.assertFalse(errors.empty(), "External serial peer must verify all three requests")
        peer_error = errors.get_nowait()
        self.assertIsNone(peer_error, f"External serial peer rejected the exchange: {peer_error}")
        report = json.loads(result.stdout)
        self.assertEqual(report["count_requested"], 3)
        self.assertEqual(report["count_verified"], 3)
        self.assertEqual(report["baudrate"], 115200)
        self.assertEqual(report["port"], port)
        self.assertTrue(math.isfinite(report["elapsed_s"]))
        self.assertGreater(report["elapsed_s"], 0)
        for statistic in ("min", "mean", "p95", "max"):
            with self.subTest(statistic=statistic):
                measured = report["roundtrip_ms"][statistic]
                self.assertTrue(math.isfinite(measured))
                self.assertGreaterEqual(measured, 0)


if __name__ == "__main__":
    unittest.main()
