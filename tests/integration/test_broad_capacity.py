"""Full H96 capacity gate with persistent evidence for failed tool runs."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "artifacts/broad-integer-v1/export"
TOOL_BIN = ROOT / ".tools/tang-toolchain/oss-cad-suite/bin"
SCRATCH = ROOT / "build/broad-capacity-scratch"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_hashes():
    return {str(path.relative_to(EXPORT)):sha256(path) for path in EXPORT.rglob("*") if path.is_file()}


class BroadCapacityTracerTests(unittest.TestCase):
    def test_public_h96_export_routes_at_50mhz_with_48_bsrams_and_bound_bitstream(self):
        before = export_hashes()
        self.assertIn("manifest.json",before,"Public H96 export must be available")
        self.assertEqual(json.loads((EXPORT/"manifest.json").read_text())["config"]["hidden_size"],96)
        for name in ("yosys","nextpnr-himbaechel","gowin_pack"):
            self.assertTrue(os.access(TOOL_BIN/name,os.X_OK),f"Real FPGA tool must exist: {name}")
        SCRATCH.mkdir(parents=True,exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="h96-heap-seed3-",dir=SCRATCH))
        print(f"Persistent H96 capacity evidence: {output}",flush=True)
        command = [str(ROOT/".venv/bin/python"),str(ROOT/"scripts/build_broad.py"),
                   "--export-dir",str(EXPORT),"--build-dir",str(output),
                   "--placer","heap","--seed","3","--heap-beta","0.9","--stage-timeout","1200"]
        result = subprocess.run(
            command,cwd=ROOT,env={**os.environ,"PATH":str(TOOL_BIN)+os.pathsep+os.environ.get("PATH","")},
            capture_output=True,text=True,timeout=3700,
        )
        (output/"test.stdout.log").write_text(result.stdout)
        (output/"test.stderr.log").write_text(result.stderr)
        (output/"test-command.json").write_text(json.dumps(command,indent=2))
        self.assertEqual(export_hashes(),before,"Capacity checking must preserve the entire frozen export")
        route_log = output/"nextpnr.log"
        diagnostic = route_log.read_text()[-5000:] if route_log.exists() else "No nextpnr log produced"
        self.assertEqual(result.returncode,0,
                         f"Public H96 must build and route legally; evidence retained in {output}\n"
                         +result.stdout+result.stderr+"\nLast placement/routing diagnostics:\n"+diagnostic)
        report = json.loads((output/"report.json").read_text())
        self.assertEqual(report["top"],"broad_top")
        self.assertEqual(report["hidden_size"],96)
        self.assertEqual(report["clock_mhz"],50)
        self.assertEqual(report["placement"],{"placer":"heap","seed":3,"heap_beta":.9})
        self.assertEqual(report["router"],"router2")
        self.assertEqual(report["stage_timeout_s"],1200)
        self.assertEqual(report["timing"]["constrained_mhz"],50)
        self.assertTrue(math.isfinite(report["timing"]["achieved_mhz"]))
        self.assertGreaterEqual(report["timing"]["achieved_mhz"],50)
        self.assertEqual(report["utilization"]["BSRAM"]["used"],48)
        self.assertEqual(sum(memory["block_count"] for memory in report["memories"].values()),48)
        self.assertEqual(report["export"]["manifest_sha256"],before["manifest.json"])
        self.assertEqual(report["bitstream"]["file"],"broad.fs")
        bitstream = output/report["bitstream"]["file"]
        self.assertTrue(bitstream.is_file())
        self.assertGreater(bitstream.stat().st_size,0)
        self.assertEqual(report["bitstream"]["sha256"],sha256(bitstream))


if __name__=="__main__":
    unittest.main()
