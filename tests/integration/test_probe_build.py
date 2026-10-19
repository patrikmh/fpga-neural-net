"""Actual FPGA build tracer; never programs connected hardware."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SOURCES = (
    "rtl/uart.sv",
    "rtl/probe_top.sv",
    "rtl/probe_rom.hex",
    "constraints/primer25k.cst",
)


class ProbeBuildTracerTests(unittest.TestCase):
    def test_probe_build_produces_verified_50mhz_bitstream_with_block_rom_and_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = subprocess.run(
                ["make", "probe-build", f"BUILD_DIR={output}"],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(
                result.returncode, 0,
                "Actual FPGA build must succeed:\n" + result.stdout + result.stderr,
            )
            report_path = output / "report.json"
            self.assertTrue(report_path.is_file(), "Build must write report.json")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["top"], "probe_top")
            self.assertEqual(report["device"], "GW5A-LV25MG121NC1/I0")
            self.assertEqual(report["clock_mhz"], 50)

            origins = {}
            for memory in ("rom", "ram"):
                with self.subTest(memory=memory):
                    evidence = report["memories"][memory]
                    self.assertGreater(evidence["block_count"], 0)
                    self.assertIsInstance(evidence["cell_types"], list)
                    self.assertTrue(evidence["cell_types"])
                    self.assertTrue(all(isinstance(item, str) and item
                                        for item in evidence["cell_types"]))
                    self.assertIsInstance(evidence["origins"], list)
                    self.assertTrue(evidence["origins"])
                    self.assertTrue(all(isinstance(item, str) and item
                                        for item in evidence["origins"]))
                    origins[memory] = set(evidence["origins"])
            self.assertTrue(
                origins["rom"].isdisjoint(origins["ram"]),
                "ROM and RAM must have distinct mapped block-memory origins",
            )
            self.assertEqual(report["timing"]["constrained_mhz"], 50)
            self.assertGreaterEqual(report["timing"]["achieved_mhz"], 50)

            filename = report["bitstream"]["file"]
            self.assertIsInstance(filename, str)
            self.assertEqual(Path(filename).name, filename)
            bitstream = (output / filename).resolve()
            self.assertEqual(bitstream.parent, output.resolve())
            self.assertTrue(bitstream.is_file(), "Reported bitstream must exist")
            self.assertGreater(bitstream.stat().st_size, 0)
            self.assertEqual(
                report["bitstream"]["sha256"],
                hashlib.sha256(bitstream.read_bytes()).hexdigest(),
            )
            for filename in BUILD_SOURCES:
                with self.subTest(source=filename):
                    self.assertEqual(
                        report["sources"][filename],
                        hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                    )
            for name in ("yosys", "nextpnr-himbaechel", "gowin_pack"):
                with self.subTest(tool=name):
                    identity = report["tools"][name]
                    self.assertIsInstance(identity, str)
                    self.assertTrue(identity.strip(), "Build tool identity must be recorded")


if __name__ == "__main__":
    unittest.main()
