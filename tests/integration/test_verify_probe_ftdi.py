"""CLI tracer loading a compiled external libftdi device fixture."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class VerifyProbeFtdiTracerTests(unittest.TestCase):
    def test_usb_verification_configures_drains_and_closes_libftdi_device(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            library = workspace / "fake_ftdi.dylib"
            trace = workspace / "libftdi.trace"
            compiled = subprocess.run(
                [shutil.which("clang") or "clang", "-shared", "-fPIC", "-O0",
                 "tests/integration/fixtures/fake_ftdi.c", "-o", str(library)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-m", "host.verify_probe",
                 "--usb-serial", "2023112818", "--libftdi", str(library), "--count", "3"],
                cwd=ROOT, env={**os.environ, "FAKE_FTDI_TRACE": str(trace)},
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(
                result.returncode, 0,
                "USB verification CLI must complete the external libftdi exchange:\n"
                + result.stdout + result.stderr,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["transport"], "libftdi")
            self.assertEqual(report["usb_serial"], "2023112818")
            self.assertEqual(report["usb_requested_baudrate"], 460800)
            self.assertEqual(report["baudrate"], 115200)
            self.assertEqual(report["count_requested"], 3)
            self.assertEqual(report["count_verified"], 3)
            self.assertTrue(math.isfinite(report["elapsed_s"]))
            self.assertGreater(report["elapsed_s"], 0)
            for statistic in ("min", "mean", "p95", "max"):
                with self.subTest(statistic=statistic):
                    value = report["roundtrip_ms"][statistic]
                    self.assertTrue(math.isfinite(value) and value >= 0)
            events = trace.read_text().splitlines()
            self.assertFalse(any(event.startswith("ERROR") for event in events), events)
            for event in ("new", "interface B", "open 0403:6010 2023112818", "bitmode 0",
                          "requested baudrate 460800", "line 8N1", "flow 0", "latency 2"):
                self.assertIn(event, events)
            self.assertLess(events.index("drain stale"), events.index("drain quiet"))
            self.assertLess(events.index("drain quiet"), events.index("request verified"))
            self.assertEqual(events.count("request verified"), 3)
            self.assertEqual(events.count("response fragment"), 15)
            self.assertEqual(events[-2:], ["close", "free"])


# Transport fault behaviors:
# 1. Unsupported ABIs are rejected before allocating a native context.
# 2. Open failures free the allocated context without closing an unopened device.
# 3. Native setup/read failures close and free an opened device.
# 4. Startup data that never becomes quiet causes a bounded failure.
# 5. A silent device causes a bounded reply timeout.
# 6. Repeated zero-byte writes cause a bounded failure.
class VerifyProbeFtdiFaultTests(unittest.TestCase):
    def rejected_exchange(self, fault, diagnostic):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            library = workspace / "fake_ftdi.dylib"
            trace = workspace / "libftdi.trace"
            compiled = subprocess.run(
                [shutil.which("clang") or "clang", "-shared", "-fPIC", "-O0",
                 "tests/integration/fixtures/fake_ftdi.c", "-o", str(library)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-m", "host.verify_probe",
                 "--usb-serial", "2023112818", "--libftdi", str(library), "--count", "1"],
                cwd=ROOT,
                env={**os.environ, "FAKE_FTDI_TRACE": str(trace), "FAKE_FTDI_FAULT": fault},
                capture_output=True, text=True, timeout=5,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "", "Failed verification must not emit success JSON")
            self.assertIn(diagnostic.lower(), result.stderr.lower())
            return trace.read_text().splitlines()

    def test_unsupported_abi_is_rejected_before_context_allocation(self):
        events = self.rejected_exchange("abi", "unsupported libftdi ABI")
        self.assertEqual(events, ["unsupported ABI"])

    def test_open_failure_frees_context_without_closing_device(self):
        events = self.rejected_exchange("open", "ftdi_usb_open_desc")
        self.assertIn("new", events)
        self.assertNotIn("close", events)
        self.assertEqual(events[-1], "free")

    def test_native_setup_or_read_failure_closes_and_frees_device(self):
        for fault, diagnostic in (("setup", "ftdi_set_bitmode"), ("read", "ftdi_read_data")):
            with self.subTest(operation=fault):
                events = self.rejected_exchange(fault, diagnostic)
                self.assertEqual(events[-2:], ["close", "free"])

    def test_startup_that_never_becomes_quiet_fails_within_deadline(self):
        events = self.rejected_exchange("never_quiet", "drain")
        self.assertIn("never quiet", events)
        self.assertNotIn("request verified", events)
        self.assertEqual(events[-2:], ["close", "free"])

    def test_silent_response_times_out_and_releases_device(self):
        events = self.rejected_exchange("silent", "timeout")
        self.assertEqual(events.count("request verified"), 1)
        self.assertEqual(events[-2:], ["close", "free"])

    def test_zero_byte_writes_fail_within_deadline_and_release_device(self):
        events = self.rejected_exchange("zero_write", "write")
        self.assertNotIn("request verified", events)
        self.assertEqual(events[-2:], ["close", "free"])


if __name__ == "__main__":
    unittest.main()
