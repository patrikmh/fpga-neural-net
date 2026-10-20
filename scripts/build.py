"""Build the UART probe and verify routed memory and timing evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "rtl/uart.sv", "rtl/probe_top.sv", "rtl/probe_rom.hex",
    "constraints/primer25k.cst", "scripts/build.py",
)
DEVICE = "GW5A-LV25MG121NC1/I0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stage(name, command, output):
    log = output / f"{name}.log"
    print(f"{name}: running (log: {log})", flush=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=240
        )
    if completed.returncode:
        raise RuntimeError(f"{name} failed with status {completed.returncode}; see {log}")


def memory_evidence(routed):
    cells = routed["modules"]["top"]["cells"]
    memories = {}
    for memory, cell_type in (("rom", "SP"), ("ram", "SDP")):
        matches = {
            name: cell for name, cell in cells.items()
            if re.fullmatch(rf"{memory}\.\d+\.\d+(?:_AUX)?", name)
            and cell["type"] == cell_type
            and cell.get("attributes", {}).get("NEXTPNR_BEL", "").endswith("/BSRAM")
        }
        if not matches:
            raise RuntimeError(f"No recognized routed block memory for {memory}")
        memories[memory] = {
            "block_count": len(matches),
            "cell_types": sorted({cell["type"] for cell in matches.values()}),
            "origins": sorted(matches),
        }
    if set(memories["rom"]["origins"]) & set(memories["ram"]["origins"]):
        raise RuntimeError("ROM and RAM origins overlap")
    return memories


def timing_evidence(timing):
    clocks = timing["fmax"]
    if not clocks:
        raise RuntimeError("Routed timing report contains no clocks")
    for clock in clocks.values():
        if clock["constraint"] != 50 or not math.isfinite(clock["achieved"]):
            raise RuntimeError("Unrecognized or incorrect clock timing")
        if clock["achieved"] < 50:
            raise RuntimeError("Routed design does not meet 50 MHz")
    return {
        "constrained_mhz": 50,
        "achieved_mhz": min(clock["achieved"] for clock in clocks.values()),
    }


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/probe")
    parser.add_argument("--uart-divisor", type=positive_integer, default=434)
    args = parser.parse_args()
    output = args.build_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    tools = {}
    identities = {}
    for name in ("yosys", "nextpnr-himbaechel", "gowin_pack"):
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f"Required tool is missing: {name}")
        tools[name] = path
        if name == "gowin_pack":
            identities[name] = f"{path} sha256:{sha256(Path(path))}"
        else:
            result = subprocess.run(
                [path, "-V" if name == "yosys" else "--version"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            identities[name] = (result.stdout + result.stderr).strip()
            if not identities[name]:
                raise RuntimeError(f"Tool identity is empty: {name}")
    synth = output / "synth.json"
    routed = output / "routed.json"
    timing_path = output / "timing.json"
    bitstream = output / "probe.fs"
    run_stage("yosys", [tools["yosys"], "-p",
        "read_verilog -sv rtl/uart.sv rtl/probe_top.sv; "
        f"chparam -set UART_DIVISOR {args.uart_divisor} probe_top; "
        "synth_gowin -top probe_top -family gw5a -nolutram; "
        f"write_json {json.dumps(str(synth))}"], output)
    run_stage("nextpnr", [tools["nextpnr-himbaechel"],
        "--json", str(synth), "--device", DEVICE,
        "--vopt", "family=GW5A-25A", "--vopt", f"cst={ROOT / SOURCES[3]}",
        "--freq", "50", "--seed", "1", "--report", str(timing_path),
        "--write", str(routed)], output)
    memories = memory_evidence(json.loads(routed.read_text()))
    raw_timing = json.loads(timing_path.read_text())
    timing = timing_evidence(raw_timing)
    if sum(item["block_count"] for item in memories.values()) != raw_timing["utilization"]["BSRAM"]["used"]:
        raise RuntimeError("Routed memory origins disagree with BSRAM utilization")
    run_stage("gowin_pack", [tools["gowin_pack"], "-d", "GW5A-25A",
        "-o", str(bitstream), str(routed)], output)
    if not bitstream.is_file() or not bitstream.stat().st_size:
        raise RuntimeError("Packer did not produce a nonempty bitstream")
    if source_hashes != {name: sha256(ROOT / name) for name in SOURCES}:
        raise RuntimeError("Build sources changed while the build was running")
    report = {
        "top": "probe_top", "device": DEVICE, "clock_mhz": 50,
        "uart_divisor": args.uart_divisor,
        "expected_fpga_baudrate": 50_000_000 / args.uart_divisor,
        "memories": memories, "timing": timing,
        "utilization": raw_timing["utilization"],
        "bitstream": {"file": bitstream.name, "sha256": sha256(bitstream)},
        "sources": source_hashes, "tools": identities,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified build: {timing['achieved_mhz']:.2f} MHz; report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
