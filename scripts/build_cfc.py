"""Build the frozen CfC export and verify routed ROM and clock evidence."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.integer import GoldenIntegerPredictor
from scripts.build import DEVICE, run_stage, sha256, timing_evidence


SOURCES = ("rtl/uart.sv", "rtl/cfc_core.sv", "rtl/cfc_top.sv", "constraints/primer25k.cst")
ROMS = ("weights.hex", "tanh.hex", "sigmoid.hex")


def memory_evidence(routed):
    cells = routed["modules"]["top"]["cells"]
    memories = {}
    origins = set()
    for memory, prefix, expected in (("weights", "weights", 1),
                                     ("tanh", "tanh_rom", 4),
                                     ("sigmoid", "sigmoid_rom", 3)):
        matches = {
            name: cell for name, cell in cells.items()
            if re.fullmatch(rf"core\.{prefix}\.\d+\.\d+(?:_AUX)?", name)
            and cell["type"] == "SP"
            and cell.get("attributes", {}).get("NEXTPNR_BEL", "").endswith("/BSRAM")
        }
        if len(matches) != expected or origins.intersection(matches):
            raise RuntimeError(f"Unrecognized routed {memory} block memory mapping: {sorted(matches)}")
        origins.update(matches)
        memories[memory] = {
            "block_count": len(matches), "cell_types": sorted({cell["type"] for cell in matches.values()}),
            "origins": sorted(matches),
        }
    return memories


def build(export_dir, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Build output directory must be empty: {output}")
    GoldenIntegerPredictor.from_export(export_dir)
    if output == export_dir or export_dir in output.parents:
        raise ValueError("Build output must be outside the frozen export")
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    export_hashes = {str(path.relative_to(export_dir)): sha256(path)
                     for path in export_dir.rglob("*") if path.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    tools, identities = {}, {}
    for name in ("yosys", "nextpnr-himbaechel", "gowin_pack"):
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f"Required tool is missing: {name}")
        tools[name] = path
        if name == "gowin_pack":
            identities[name] = f"{path} sha256:{sha256(Path(path))}"
        else:
            result = subprocess.run([path, "-V" if name == "yosys" else "--version"],
                                    capture_output=True, text=True, check=True, timeout=10)
            identities[name] = (result.stdout + result.stderr).strip()
            if not identities[name]:
                raise RuntimeError(f"Tool identity is empty: {name}")
    synth, routed = output / "synth.json", output / "routed.json"
    timing_path, bitstream = output / "timing.json", output / "cfc.fs"
    parameters = " ".join(f"-set {parameter} {json.dumps(str(export_dir / filename))}"
                          for parameter, filename in zip(("WEIGHTS_FILE", "TANH_FILE", "SIGMOID_FILE"), ROMS))
    run_stage("yosys", [tools["yosys"], "-p",
        "read_verilog -sv rtl/uart.sv rtl/cfc_core.sv rtl/cfc_top.sv; "
        f"chparam {parameters} cfc_top; "
        "synth_gowin -top cfc_top -family gw5a -nolutram; "
        f"write_json {json.dumps(str(synth))}"], output)
    run_stage("nextpnr", [tools["nextpnr-himbaechel"], "--json", str(synth),
        "--device", DEVICE, "--vopt", "family=GW5A-25A",
        "--vopt", f"cst={ROOT / SOURCES[3]}", "--freq", "50", "--seed", "1",
        "--report", str(timing_path), "--write", str(routed)], output)
    memories = memory_evidence(json.loads(routed.read_text()))
    raw_timing = json.loads(timing_path.read_text())
    timing = timing_evidence(raw_timing)
    if raw_timing["utilization"]["BSRAM"]["used"] != 8:
        raise RuntimeError("Mapped ROM origins disagree with routed BSRAM utilization")
    run_stage("gowin_pack", [tools["gowin_pack"], "-d", "GW5A-25A",
        "-o", str(bitstream), str(routed)], output)
    if not bitstream.is_file() or not bitstream.stat().st_size:
        raise RuntimeError("Packer did not produce a nonempty bitstream")
    if source_hashes != {name: sha256(ROOT / name) for name in SOURCES}:
        raise RuntimeError("RTL or constraints changed during build")
    if export_hashes != {str(path.relative_to(export_dir)): sha256(path)
                         for path in export_dir.rglob("*") if path.is_file()}:
        raise RuntimeError("Frozen export changed during build")
    report = {
        "top": "cfc_top", "device": DEVICE, "clock_mhz": 50,
        "memories": memories, "timing": timing, "utilization": raw_timing["utilization"],
        "sources": source_hashes, "tools": identities,
        "bitstream": {"file": bitstream.name, "sha256": sha256(bitstream)},
        "export": {"manifest_sha256": export_hashes["manifest.json"],
                   "files": {name: export_hashes[name] for name in ROMS}},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified CfC build: {timing['achieved_mhz']:.2f} MHz; report: {output / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "build/integer-v1")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/cfc")
    args = parser.parse_args()
    build(args.export_dir.resolve(), args.build_dir.resolve())


if __name__ == "__main__":
    main()
