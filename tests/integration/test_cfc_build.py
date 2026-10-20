"""Real CfC synthesis/place-route/pack tracer without hardware programming."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "artifacts/cfc8-q13-v1"
ROMS = ("weights.hex", "tanh.hex", "sigmoid.hex")
SOURCES = ("rtl/uart.sv", "rtl/cfc_core.sv", "rtl/cfc_top.sv", "constraints/primer25k.cst")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_hashes():
    return {str(path.relative_to(EXPORT)): sha256(path)
            for path in EXPORT.rglob("*") if path.is_file()}


class CfcBuildTracerTests(unittest.TestCase):
    def test_nonempty_build_output_is_rejected_before_tools_and_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinels = {"report.json": b'{"stale": true}\n', "cfc.fs": b"old bitstream\n"}
            for filename, contents in sentinels.items():
                (output / filename).write_bytes(contents)

            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/build_cfc.py"),
                 "--export-dir", str(EXPORT), "--build-dir", str(output)],
                cwd=ROOT, env={**os.environ, "PATH": ""},
                capture_output=True, text=True, timeout=10,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("empty", result.stderr.lower(),
                          "Nonempty output must be rejected before checking FPGA tools")
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, sentinels)

    def test_cfc_build_produces_traceable_50mhz_bitstream_with_eight_distinct_rom_blocks(self):
        before = export_hashes()
        self.assertIn("manifest.json", before, "Approved frozen export must be present")
        sigmoid_words = [int(word, 16) for word in (EXPORT / "sigmoid.hex").read_text().split()]
        self.assertEqual(len(sigmoid_words), 4096)
        self.assertTrue(all(value & 0xE000 == 0 for value in sigmoid_words),
                        "Constant-zero upper bits allow the sigmoid ROM to use 13-bit storage")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = subprocess.run(
                ["make", "cfc-build", f"EXPORT_DIR={EXPORT}", f"CFC_BUILD_DIR={output}"],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(result.returncode, 0,
                             "Actual CfC FPGA build must succeed:\n" + result.stdout + result.stderr)
            report_path = output / "report.json"
            self.assertTrue(report_path.is_file(), "Build must emit report.json")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["top"], "cfc_top")
            self.assertEqual(report["device"], "GW5A-LV25MG121NC1/I0")
            self.assertEqual(report["clock_mhz"], 50)
            self.assertEqual(report["timing"]["constrained_mhz"], 50)
            achieved = report["timing"]["achieved_mhz"]
            self.assertTrue(math.isfinite(achieved))
            self.assertGreaterEqual(achieved, 50)

            all_origins = set()
            mapped_count = 0
            for memory, expected_blocks in (("weights", 1), ("tanh", 4), ("sigmoid", 3)):
                with self.subTest(memory=memory):
                    mapping = report["memories"][memory]
                    self.assertEqual(mapping["block_count"], expected_blocks)
                    self.assertIsInstance(mapping["cell_types"], list)
                    self.assertTrue(mapping["cell_types"])
                    self.assertTrue(all(isinstance(name, str) and name for name in mapping["cell_types"]))
                    self.assertIsInstance(mapping["origins"], list)
                    self.assertEqual(len(mapping["origins"]), expected_blocks)
                    self.assertTrue(all(isinstance(name, str) and name for name in mapping["origins"]))
                    origins = set(mapping["origins"])
                    self.assertEqual(len(origins), expected_blocks)
                    self.assertTrue(all_origins.isdisjoint(origins), "ROMs must map to distinct block origins")
                    all_origins.update(origins)
                    mapped_count += mapping["block_count"]
            self.assertEqual(mapped_count, 8)
            self.assertEqual(report["utilization"]["BSRAM"]["used"], mapped_count)
            self.assertGreaterEqual(report["utilization"]["BSRAM"]["available"], mapped_count)

            filename = report["bitstream"]["file"]
            self.assertIsInstance(filename, str)
            self.assertEqual(Path(filename).name, filename)
            bitstream = (output / filename).resolve()
            self.assertEqual(bitstream.parent, output.resolve())
            self.assertTrue(bitstream.is_file())
            self.assertGreater(bitstream.stat().st_size, 0)
            self.assertEqual(report["bitstream"]["sha256"], sha256(bitstream))
            for filename in SOURCES:
                self.assertEqual(report["sources"][filename], sha256(ROOT / filename))
            self.assertEqual(report["export"]["manifest_sha256"], before["manifest.json"])
            for filename in ROMS:
                self.assertEqual(report["export"]["files"][filename], before[filename])
            for tool in ("yosys", "nextpnr-himbaechel", "gowin_pack"):
                self.assertIsInstance(report["tools"][tool], str)
                self.assertTrue(report["tools"][tool].strip())
            self.assertEqual(export_hashes(), before, "Building must preserve the entire frozen export")


if __name__ == "__main__":
    unittest.main()
