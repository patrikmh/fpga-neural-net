"""CfC host verification CLI through a real ctypes-loaded USB boundary."""

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from model.integer import GoldenIntegerPredictor


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "artifacts/cfc8-q13-v1"
VECTORS = EXPORT / "vectors"


def frame(payload):
    checksum = 0
    for value in payload:
        checksum ^= value
    return payload + bytes([checksum])


def provenance():
    return {str(path.relative_to(EXPORT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in EXPORT.rglob("*") if path.is_file()}


class VerifyCfcFtdiTracerTests(unittest.TestCase):
    def test_usb_cli_resets_then_verifies_three_complete_cfc_vectors_and_reports_timings(self):
        before = provenance()
        predictor = GoldenIntegerPredictor.from_export(EXPORT)
        rows = (VECTORS / "vectors.hex").read_text().splitlines()[:3]
        requests = [frame(b"\xa5\x5a" + struct.pack("<HBh", 65535, 2, 0))]
        replies = [frame(b"\x5a\xa5" + struct.pack("<HBh8hI", 65535, 0, 0, *([0] * 8), 0))]
        for sequence, row in enumerate(rows):
            words = [int(row[offset:offset + 4], 16) for offset in range(0, 40, 4)]
            signed = [value - (65536 if value >= 32768 else 0) for value in words]
            hidden, y = predictor.step(signed[0])
            self.assertEqual(hidden + [y], signed[1:], "Committed vectors must match checked golden model")
            requests.append(frame(b"\xa5\x5a" + struct.pack("<HBh", sequence, 1, signed[0])))
            replies.append(frame(b"\x5a\xa5" + struct.pack("<HBh8hI", sequence, 0, y, *hidden, 1052)))
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            library, trace = workspace / "fake_cfc.dylib", workspace / "trace.txt"
            request_path, reply_path = workspace / "requests.bin", workspace / "replies.bin"
            request_path.write_bytes(b"".join(requests))
            reply_path.write_bytes(b"".join(replies))
            compiled = subprocess.run(
                [shutil.which("clang") or "clang", "-shared", "-fPIC", "-O0",
                 "tests/integration/fixtures/fake_cfc_ftdi.c", "-o", str(library)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            output = workspace / "report.json"
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-m", "host.verify_cfc", "--usb-serial", "2023112818",
                 "--export-dir", str(EXPORT), "--vectors-dir", str(VECTORS), "--count", "3",
                 "--libftdi", str(library), "--output", str(output)],
                cwd=ROOT, env={**os.environ, "FAKE_CFC_TRACE": str(trace),
                               "FAKE_CFC_REQUESTS": str(request_path), "FAKE_CFC_REPLIES": str(reply_path)},
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0,
                             "Real CfC USB verification CLI must complete:\n" + result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(json.loads(result.stdout), report)
            self.assertEqual(report["count_requested"], 3)
            self.assertEqual(report["count_verified"], 3)
            self.assertEqual(report["transport"], "libftdi")
            self.assertEqual(report["usb_serial"], "2023112818")
            self.assertEqual(report["usb_requested_baudrate"], 460800)
            self.assertEqual(report["baudrate"], 115200)
            self.assertEqual(report["export_manifest_sha256"], before["manifest.json"])
            self.assertEqual(report["vectors_manifest_sha256"], before["vectors/manifest.json"])
            manifest = json.loads((EXPORT / "manifest.json").read_text())
            self.assertEqual(report["checkpoint_sha256"], manifest["checkpoint_sha256"])
            self.assertTrue(math.isfinite(report["elapsed_s"]) and report["elapsed_s"] > 0)
            for name in ("min", "mean", "p95", "max"):
                with self.subTest(statistic=name):
                    self.assertAlmostEqual(report["compute_us"][name], 1052 / 50)
                    self.assertTrue(math.isfinite(report["roundtrip_ms"][name]))
                    self.assertGreaterEqual(report["roundtrip_ms"][name], 0)
            events = trace.read_text().splitlines()
            self.assertFalse(any(event.startswith("ERROR") for event in events), events)
            self.assertLess(events.index("drain stale"), events.index("drain quiet"))
            self.assertLess(events.index("drain quiet"), events.index("request RESET"))
            self.assertEqual(events.count("request RESET"), 1)
            self.assertEqual(events.count("request STEP"), 3)
            self.assertGreater(events.count("response fragment"), 4)
            self.assertEqual(events[-2:], ["close", "free"])
            self.assertEqual(provenance(), before)


# Additional behaviors: malformed STEP rejection; invalid RESET rejection;
# absent/short reply timeout; output-path isolation from frozen artifacts.
class VerifyCfcFtdiFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.library = Path(temporary.name) / "fake_cfc.dylib"
        compiled = subprocess.run(
            [shutil.which("clang") or "clang", "-shared", "-fPIC", "-O0",
             "tests/integration/fixtures/fake_cfc_ftdi.c", "-o", str(cls.library)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        if compiled.returncode:
            raise AssertionError(compiled.stdout + compiled.stderr)

    def exchange(self, workspace, mutation=None, fault="", export=EXPORT, output=None, live=None):
        row = (export / "vectors/vectors.hex").read_text().splitlines()[0]
        words = [int(row[offset:offset + 4], 16) for offset in range(0, 40, 4)]
        signed = [value - (65536 if value >= 32768 else 0) for value in words]
        hidden, y = GoldenIntegerPredictor.from_export(export).step(signed[0])
        self.assertEqual(hidden + [y], signed[1:])
        requests = [frame(b"\xa5\x5a" + struct.pack("<HBh", 65535, 2, 0)),
                    frame(b"\xa5\x5a" + struct.pack("<HBh", 0, 1, signed[0]))]
        replies = [frame(b"\x5a\xa5" + struct.pack("<HBh8hI", 65535, 0, 0, *([0] * 8), 0)),
                   frame(b"\x5a\xa5" + struct.pack("<HBh8hI", 0, 0, y, *hidden, 1052))]
        if mutation is not None:
            index, offset, value, repair_checksum = mutation
            changed = bytearray(replies[index])
            if offset == 23:
                struct.pack_into("<I", changed, offset, value)
            else:
                changed[offset] ^= value
            replies[index] = frame(bytes(changed[:-1])) if repair_checksum else bytes(changed)
        request_path, reply_path = workspace / "requests.bin", workspace / "replies.bin"
        request_path.write_bytes(b"".join(requests))
        reply_path.write_bytes(b"".join(replies))
        trace = workspace / "trace.txt"
        output = output if output is not None else workspace / "report.json"
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-m", "host.verify_cfc", "--usb-serial", "2023112818",
             "--export-dir", str(export), "--vectors-dir", str(export / "vectors"), "--count", "1",
             "--libftdi", str(self.library), "--output", str(output),
             *(["--live-dir", str(live)] if live is not None else [])],
            cwd=ROOT, env={**os.environ, "FAKE_CFC_TRACE": str(trace),
                           "FAKE_CFC_REQUESTS": str(request_path), "FAKE_CFC_REPLIES": str(reply_path),
                           "FAKE_CFC_FAULT": fault},
            capture_output=True, text=True, timeout=5,
        )
        return result, output, trace.read_text().splitlines() if trace.exists() else []

    def assert_rejected(self, result, output, events, diagnostic):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "", "Failure must not emit successful JSON")
        self.assertFalse(output.exists(), "Failure must not create a successful report")
        self.assertRegex(result.stderr.lower(), diagnostic)
        self.assertEqual(events[-2:], ["close", "free"])

    def test_invalid_complete_step_replies_are_rejected_and_release_device(self):
        cases = {
            "crc": ((1, 27, 1, False), "checksum"),
            "sequence": ((1, 2, 1, True), "sequence"),
            "output": ((1, 5, 1, True), "mismatch"),
            "hidden": ((1, 7, 1, True), "mismatch"),
            "status": ((1, 4, 1, True), "status"),
            "zero_cycles": ((1, 23, 0, True), "cycle"),
            "excess_cycles": ((1, 23, 5000, True), "cycle"),
        }
        for name, (mutation, diagnostic) in cases.items():
            with self.subTest(cause=name), tempfile.TemporaryDirectory() as directory:
                result, output, events = self.exchange(Path(directory), mutation=mutation)
                self.assert_rejected(result, output, events, diagnostic)

    def test_report_persistence_failure_marks_live_status_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            blocker = workspace / "blocker"
            blocker.write_text("regular file prevents report directory creation")
            live = workspace / "live"
            result, output, events = self.exchange(
                workspace, output=blocker / "report.json", live=live,
            )
            self.assert_rejected(result, output, events, r"(directory|file|exist)")
            latest = json.loads((live / "latest.json").read_text())
            self.assertEqual(latest["status"], "failed")

    def test_nonzero_reset_state_output_or_cycles_are_rejected(self):
        for offset in (5, 7, 23):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                result, output, events = self.exchange(Path(directory), mutation=(0, offset, 1, True))
                self.assert_rejected(result, output, events, "reset")

    def test_silent_or_short_step_replies_time_out_and_release_device(self):
        for fault in ("silent", "short"):
            with self.subTest(reply=fault), tempfile.TemporaryDirectory() as directory:
                result, output, events = self.exchange(Path(directory), fault=fault)
                self.assert_rejected(result, output, events, "timeout")

    def test_output_inside_frozen_export_or_vectors_is_rejected_before_usb(self):
        for relative in ("report.json", "vectors/report.json"):
            with self.subTest(output=relative), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                copied = workspace / "exported"
                shutil.copytree(EXPORT, copied)
                before = {str(path.relative_to(copied)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in copied.rglob("*") if path.is_file()}
                result, output, events = self.exchange(workspace, export=copied, output=copied / relative)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.strip(), "")
                self.assertRegex(result.stderr.lower(), r"(output|frozen|outside|artifact)")
                self.assertEqual(events, [], "Frozen-output guard must run before USB allocation")
                self.assertFalse(output.exists())
                after = {str(path.relative_to(copied)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in copied.rglob("*") if path.is_file()}
                self.assertEqual(after, before)


# Demo behaviors: completed bounded telemetry for count mode, and timed continuous
# input cycling with one initial reset and genuine elapsed-time acceptance gates.
class VerifyCfcLiveDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.library = Path(temporary.name) / "fake_cfc.dylib"
        compiled = subprocess.run(
            [shutil.which("clang") or "clang", "-shared", "-fPIC", "-O0",
             "tests/integration/fixtures/fake_cfc_ftdi.c", "-o", str(cls.library)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        if compiled.returncode:
            raise AssertionError(compiled.stdout + compiled.stderr)

    def demo(self, workspace, timed):
        export = workspace / "exported"
        shutil.copytree(EXPORT, export)
        vector_path = export / "vectors/vectors.hex"
        first_three = vector_path.read_text().splitlines()[:3]
        if timed:
            vector_path.write_text("\n".join(first_three) + "\n")
            path = export / "vectors/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["count"] = 3
            manifest["vectors_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
            path.write_text(json.dumps(manifest))
        values = []
        for row in first_three:
            word = int(row[:4], 16)
            values.append(word - (65536 if word >= 32768 else 0))
        predictor = GoldenIntegerPredictor.from_export(export)
        requests = [frame(b"\xa5\x5a" + struct.pack("<HBh", 65535, 2, 0))]
        replies = [frame(b"\x5a\xa5" + struct.pack("<HBh8hI", 65535, 0, 0, *([0] * 8), 0))]
        expected = []
        for index in range(64 if timed else 3):
            x = values[index % 3]
            hidden, y = predictor.step(x)
            expected.append({"index": index, "input": x, "y": y, "golden_y": y,
                             "hidden": hidden, "cycles": 1052})
            requests.append(frame(b"\xa5\x5a" + struct.pack("<HBh", index, 1, x)))
            replies.append(frame(b"\x5a\xa5" + struct.pack("<HBh8hI", index, 0, y, *hidden, 1052)))
        request_path, reply_path = workspace / "requests.bin", workspace / "replies.bin"
        request_path.write_bytes(b"".join(requests))
        reply_path.write_bytes(b"".join(replies))
        live, output, trace = workspace / "live", workspace / "report.json", workspace / "trace.txt"
        mode = ["--duration", "0.12"] if timed else ["--count", "3"]
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-m", "host.verify_cfc", "--usb-serial", "2023112818",
             "--export-dir", str(export), "--vectors-dir", str(export / "vectors"),
             "--libftdi", str(self.library), "--output", str(output), "--live-dir", str(live), *mode],
            cwd=ROOT, env={**os.environ, "FAKE_CFC_TRACE": str(trace),
                           "FAKE_CFC_REQUESTS": str(request_path), "FAKE_CFC_REPLIES": str(reply_path),
                           "FAKE_CFC_DELAY_MS": "10" if timed else "0"},
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0,
                         "Real live/demo verification CLI must complete:\n" + result.stdout + result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(json.loads(result.stdout), report)
        latest = json.loads((live / "latest.json").read_text())
        self.assertEqual(latest["status"], "completed")
        count = report["count_verified"]
        self.assertEqual(latest["count_verified"], count)
        self.assertGreater(len(latest["samples"]), 0)
        self.assertLessEqual(len(latest["samples"]), 512)
        for sample in latest["samples"]:
            with self.subTest(sample=sample["index"]):
                self.assertLess(sample["index"], count)
                reference = expected[sample["index"]]
                self.assertEqual({name: sample[name] for name in reference}, reference)
                self.assertTrue(math.isfinite(sample["roundtrip_ms"]))
                self.assertGreaterEqual(sample["roundtrip_ms"], 0)
        for record in (latest, report):
            self.assertTrue(math.isfinite(record["throughput_hz"]))
            self.assertGreater(record["throughput_hz"], 0)
        events = trace.read_text().splitlines()
        self.assertEqual(events.count("request RESET"), 1)
        self.assertEqual(events.count("request STEP"), count)
        self.assertEqual(events[-2:], ["close", "free"])
        self.assertFalse(report["acceptance_600s"])
        return report, latest

    def test_count_mode_writes_completed_bounded_live_samples_and_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            report, latest = self.demo(Path(directory), timed=False)
            self.assertEqual(report["count_requested"], 3)
            self.assertEqual(report["count_verified"], 3)
            self.assertEqual([sample["index"] for sample in latest["samples"]], [0, 1, 2])

    def test_duration_mode_cycles_inputs_with_continuous_hidden_state_and_one_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            report, latest = self.demo(Path(directory), timed=True)
            self.assertEqual(report["duration_requested_s"], 0.12)
            self.assertGreaterEqual(report["duration_s"], 0.12)
            self.assertGreater(report["count_verified"], 3)
            self.assertLessEqual(report["count_verified"], 64)
            self.assertEqual(len(latest["samples"]), report["count_verified"])


if __name__ == "__main__":
    unittest.main()
