"""Report local FPGA tool availability without contacting hardware."""

import argparse
import glob
import json
import math
import platform
import shutil
import subprocess
import sys


PROBES = {
    "uv": "--version",
    "yosys": "-V",
    "nextpnr-himbaechel": "--version",
    "gowin_pack": "--help",
    "openFPGALoader": "--version",
    "iverilog": "-V",
    "vvp": "-V",
}


def positive_timeout(value):
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return timeout


def probe_tool(name, argument, timeout):
    path = shutil.which(name)
    result = {
        "path": path,
        "status": "missing",
        "probe_kind": "help" if name == "gowin_pack" else "version",
        "version": None,
        "exit_code": None,
        "detail": "Executable not found on PATH",
    }
    if path is None:
        return result
    try:
        completed = subprocess.run(
            [path, argument], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        result.update(status="timeout", detail=f"Probe exceeded {timeout:g} seconds")
        return result
    except OSError as error:
        result.update(status="failed", detail=str(error))
        return result
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    result.update(
        exit_code=completed.returncode,
        status="ok" if completed.returncode == 0 else "failed",
        detail=output or f"Probe exited with status {completed.returncode}",
    )
    if result["probe_kind"] == "version" and not output:
        result.update(status="failed", detail="Version probe returned no output")
    if result["status"] == "ok" and result["probe_kind"] == "version":
        result["version"] = output
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a JSON report")
    parser.add_argument("--timeout-seconds", type=positive_timeout, default=3.0)
    args = parser.parse_args()
    tools = {
        name: probe_tool(name, argument, args.timeout_seconds)
        for name, argument in PROBES.items()
    }
    supported = sys.version_info >= (3, 11)
    report = {
        "ready": supported and all(tool["status"] == "ok" for tool in tools.values()),
        "hardware_verified": False,
        "python": {"version": platform.python_version(), "supported": supported},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "tools": tools,
        "uart_candidates": sorted(glob.glob("/dev/cu.usbserial-*")),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Environment ready: {report['ready']}")
        print(f"Python {report['python']['version']} (supported: {supported})")
        print(f"Platform: {platform.system()} {platform.machine()}")
        for name, tool in tools.items():
            print(f"{name}: {tool['status']} — {tool['detail']}")
        print(f"Serial ports (enumerated only): {', '.join(report['uart_candidates']) or 'none'}")
        print("Hardware verified: False")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
