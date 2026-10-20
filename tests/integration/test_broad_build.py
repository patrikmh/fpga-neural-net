"""Actual tiny BROAD FPGA build and independently checked bank staging."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_broad_train import prepared_fixture


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")
TOOL_BIN = ROOT / ".tools/tang-toolchain/oss-cad-suite/bin"
SOURCES = ("rtl/uart.sv", "rtl/broad_core.sv", "rtl/broad_top.sv",
           "constraints/primer25k.cst", "scripts/build_broad.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BroadBuildTracerTests(unittest.TestCase):
    def test_real_build_stages_checked_weight_banks_and_produces_routed_50mhz_bitstream(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, trained, exported, output = (workspace/name for name in ("prepared","trained","exported","build"))
            prepared_fixture(prepared)
            commands = [
                ["model.broad_train", "--prepared-dir", str(prepared), "--output-dir", str(trained),
                 "--hidden-size", "3", "--epochs", "1", "--updates-per-epoch", "2", "--batch-size", "2",
                 "--window", "16", "--warmup", "2", "--validation-every", "1", "--seed", "17"],
                ["model.broad_export_integer", "--checkpoint", str(trained/"checkpoint.pt"), "--output-dir", str(exported)],
            ]
            for command in commands:
                result = subprocess.run([PYTHON,"-m",*command],cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            before = {str(path.relative_to(exported)):sha256(path) for path in exported.rglob("*") if path.is_file()}
            artifact = json.loads((exported/"manifest.json").read_text())
            result = subprocess.run(
                [PYTHON,str(ROOT/"scripts/build_broad.py"),"--export-dir",str(exported),"--build-dir",str(output)],
                cwd=ROOT,env={**os.environ,"PATH":str(TOOL_BIN)+os.pathsep+os.environ.get("PATH","")},
                capture_output=True,text=True,timeout=1900,
            )
            self.assertEqual(result.returncode,0,"Real BROAD FPGA builder must complete:\n"+result.stdout+result.stderr)
            report = json.loads((output/"report.json").read_text())
            self.assertEqual(report["top"],"broad_top")
            self.assertEqual(report["device"],"GW5A-LV25MG121NC1/I0")
            self.assertEqual(report["hidden_size"],3)
            self.assertEqual(report["clock_mhz"],50)
            self.assertEqual(report["placement"],{"placer":"heap","seed":1,"heap_beta":.9})
            self.assertEqual(report["stage_timeout_s"],600)
            self.assertEqual(report["timing"]["constrained_mhz"],50)
            self.assertTrue(math.isfinite(report["timing"]["achieved_mhz"]))
            self.assertGreaterEqual(report["timing"]["achieved_mhz"],50)
            for filename in SOURCES:
                self.assertEqual(report["sources"][filename],sha256(ROOT/filename))
            self.assertEqual(report["export"]["manifest_sha256"],before["manifest.json"])
            for filename in artifact["files"]:
                self.assertEqual(report["export"]["files"][filename],before[filename])

            flat = (exported/"weights.hex").read_text().split()
            bank_count = (len(flat)+1023)//1024
            self.assertEqual(len(report["banks"]),bank_count)
            reconstructed = []
            for number,bank in enumerate(report["banks"]):
                with self.subTest(bank=number):
                    self.assertEqual(Path(bank["file"]).name,f"weights_{number:03d}.hex")
                    path = (output/bank["file"]).resolve()
                    self.assertTrue(path.is_relative_to(output.resolve()))
                    self.assertEqual(bank["sha256"],sha256(path))
                    words = path.read_text().split()
                    self.assertEqual(len(words),1024)
                    self.assertTrue(all(len(word)==4 and all(c in "0123456789abcdefABCDEF" for c in word) for word in words))
                    self.assertEqual(bank["valid_words"],min(1024,len(flat)-number*1024))
                    reconstructed.extend(int(word,16) for word in words)
            self.assertEqual(reconstructed[:len(flat)],[int(word,16) for word in flat])
            self.assertEqual(reconstructed[len(flat):],[0]*(bank_count*1024-len(flat)))

            origins = set()
            mapped = 0
            for memory in ("weights","tanh","sigmoid"):
                with self.subTest(memory=memory):
                    evidence = report["memories"][memory]
                    self.assertGreater(evidence["block_count"],0)
                    self.assertTrue(evidence["cell_types"])
                    memory_origins = set(evidence["origins"])
                    self.assertEqual(len(memory_origins),evidence["block_count"])
                    self.assertTrue(all(isinstance(name,str) and name for name in memory_origins))
                    self.assertTrue(origins.isdisjoint(memory_origins))
                    origins.update(memory_origins)
                    mapped += evidence["block_count"]
            self.assertEqual(report["memories"]["weights"]["block_count"],bank_count)
            self.assertEqual(mapped,8)
            self.assertEqual(report["utilization"]["BSRAM"]["used"],mapped)
            filename = report["bitstream"]["file"]
            self.assertEqual(filename,"broad.fs")
            bitstream = (output/filename).resolve()
            self.assertTrue(bitstream.is_relative_to(output.resolve()))
            self.assertTrue(bitstream.is_file())
            self.assertGreater(bitstream.stat().st_size,0)
            self.assertEqual(report["bitstream"]["sha256"],sha256(bitstream))
            for name in ("yosys","nextpnr-himbaechel","gowin_pack"):
                self.assertIsInstance(report["tools"][name],str)
                self.assertTrue(report["tools"][name].strip())
            self.assertEqual({str(path.relative_to(exported)):sha256(path) for path in exported.rglob("*") if path.is_file()},before)


if __name__=="__main__":
    unittest.main()
