"""Verify one continuous CfC vector stream on an explicitly named USB device."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import statistics
import sys
import time

from host.cfc_protocol import build_request, parse_response
from host.ftdi import DEFAULT_LIBRARY, Ftdi
from host.verify_probe import request_count, usb_serial_number
from model.integer import GoldenIntegerPredictor, sha256


ROOT = Path(__file__).resolve().parents[1]


def positive_duration(value):
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError("Duration must be finite and positive")
    return duration


def publish_live(directory, status, samples, count, duration, error=None):
    if directory is None:
        return
    record = {"status": status, "samples": list(samples), "count_verified": count,
              "duration_s": duration, "throughput_hz": count / duration if duration > 0 else 0}
    if error is not None:
        record["error"] = error
    temporary = directory / ".latest.json.tmp"
    temporary.write_text(json.dumps(record, allow_nan=False) + "\n")
    temporary.replace(directory / "latest.json")


def snapshot(export_dir, vectors_dir, library):
    hashes = {}
    for label, directory in (("export", export_dir), ("vectors", vectors_dir)):
        for path in directory.rglob("*"):
            if path.is_file():
                hashes[f"{label}/{path.relative_to(directory)}"] = sha256(path)
    for name in ("host/verify_cfc.py", "host/ftdi.py", "host/cfc_protocol.py",
                 "host/verify_probe.py", "model/integer.py", "model/data.py", "model/export_vectors.py"):
        hashes[name] = sha256(ROOT / name)
    hashes["libftdi"] = sha256(library)
    return hashes


def summary(values):
    ordered = sorted(values)
    return {"min": ordered[0], "mean": statistics.mean(values),
            "p95": ordered[math.ceil(len(ordered) * 0.95) - 1], "max": ordered[-1]}


def exchange(connection, sequence, value, command=1):
    request = build_request(sequence, value, command)
    if connection.write(request) != len(request):
        raise ValueError(f"Incomplete CfC request write at sequence {sequence}")
    response = connection.read(28)
    if len(response) != 28:
        raise ValueError(f"CfC response timeout at sequence {sequence}: received {len(response)}/28 bytes")
    actual_sequence, y, hidden, cycles = parse_response(response)
    if actual_sequence != sequence:
        raise ValueError(f"CfC sequence mismatch: expected {sequence}, received {actual_sequence}")
    return y, hidden, cycles


def verify(args):
    if args.output is not None:
        output = args.output.resolve()
        for directory in (args.export_dir.resolve(), args.vectors_dir.resolve()):
            if output == directory or directory in output.parents:
                raise ValueError("Output must be outside frozen export and vector directories")
    if args.output is not None and args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    if args.live_dir is not None:
        live = args.live_dir.resolve()
        for directory in (args.export_dir.resolve(), args.vectors_dir.resolve()):
            if live == directory or directory in live.parents:
                raise ValueError("Live directory must be outside frozen export and vector directories")
        if live.exists() and any(live.iterdir()):
            raise ValueError("Live directory must be empty")
    predictor = GoldenIntegerPredictor.from_export(args.export_dir)
    artifact = json.loads((args.export_dir / "manifest.json").read_text())
    vectors_manifest_path = args.vectors_dir / "manifest.json"
    vectors_manifest = json.loads(vectors_manifest_path.read_text())
    if vectors_manifest["export_manifest_sha256"] != sha256(args.export_dir / "manifest.json"):
        raise ValueError("Vector/export manifest SHA256 identity mismatch")
    if vectors_manifest["checkpoint_sha256"] != artifact["checkpoint_sha256"]:
        raise ValueError("Vector/checkpoint SHA256 identity mismatch")
    if vectors_manifest["vectors_sha256"] != sha256(args.vectors_dir / "vectors.hex"):
        raise ValueError("Vectors SHA256 identity mismatch")
    if vectors_manifest["numeric"] != artifact["numeric"] or vectors_manifest["initial_hidden"] != [0] * 8:
        raise ValueError("Vector numeric format or initial hidden state mismatch")
    for name in ("seed", "generator"):
        if vectors_manifest[name] != artifact["config"][name]:
            raise ValueError(f"Vector {name} mismatch")
    if vectors_manifest["split"] != "validation" or vectors_manifest["trajectory_id"] != 0:
        raise ValueError("Vectors must describe one continuous validation trajectory")
    for name in ("model/data.py", "model/integer.py", "model/export_vectors.py"):
        if vectors_manifest["sources"].get(name) != sha256(ROOT / name):
            raise ValueError(f"Vector source SHA256 identity mismatch: {name}")
    rows = (args.vectors_dir / "vectors.hex").read_text().splitlines()
    if type(vectors_manifest["count"]) is not int or len(rows) != vectors_manifest["count"]:
        raise ValueError("Vector count mismatch")
    if not rows:
        raise ValueError("Vectors must contain at least one input")
    if args.count is not None and args.count > len(rows):
        raise ValueError("Requested count exceeds available vectors")
    decoded = []
    for row in rows[:args.count]:
        if len(row) != 40 or any(c not in "0123456789abcdefABCDEF" for c in row):
            raise ValueError("Invalid vector row: expected ten signed 16-bit hex words")
        words = [int(row[offset:offset + 4], 16) for offset in range(0, 40, 4)]
        decoded.append([word - (65536 if word >= 32768 else 0) for word in words])
    library = Path(args.libftdi) if args.libftdi else DEFAULT_LIBRARY
    before = snapshot(args.export_dir, args.vectors_dir, library)
    roundtrips, compute = [], []
    samples = deque(maxlen=512)
    stream_started = None
    duration = 0.0
    if args.live_dir is not None:
        args.live_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with Ftdi(args.usb_serial, str(library)) as connection:
            reset_y, reset_hidden, reset_cycles = exchange(connection, 65535, 0, command=2)
            if (reset_y, reset_hidden, reset_cycles) != (0, (0,) * 8, 0):
                raise ValueError("CfC RESET acknowledgement did not clear output, state and cycles")
            stream_started = time.perf_counter()
            last_publish = stream_started
            publish_live(args.live_dir, "running", samples, 0, 0)
            index = 0
            while args.duration is not None or index < args.count:
                sequence = index & 65535
                row = decoded[index % len(decoded)]
                hidden, expected_y = predictor.step(row[0])
                if index < len(decoded) and hidden + [expected_y] != row[1:]:
                    raise ValueError(f"Stored vector differs from golden model at sequence {sequence}")
                roundtrip_started = time.perf_counter()
                y, actual_hidden, cycles = exchange(connection, sequence, row[0])
                roundtrip_ms = (time.perf_counter() - roundtrip_started) * 1000
                if y != expected_y or actual_hidden != tuple(hidden):
                    raise ValueError(f"CfC state/output mismatch at sequence {sequence}")
                if not 0 < cycles < 5000:
                    raise ValueError(f"CfC computation cycle count out of range at sequence {sequence}: {cycles}")
                roundtrips.append(roundtrip_ms)
                compute.append(cycles / 50)
                samples.append({"index": index, "input": row[0], "y": y, "golden_y": expected_y,
                                "hidden": list(actual_hidden), "cycles": cycles, "roundtrip_ms": roundtrip_ms})
                index += 1
                now = time.perf_counter()
                duration = now - stream_started
                if now - last_publish >= 0.5:
                    publish_live(args.live_dir, "running", samples, index, duration)
                    last_publish = now
                if index % 100 == 0:
                    print(f"Verified {index} CfC steps ({duration:.1f}s)", file=sys.stderr)
                if args.duration is not None and duration >= args.duration:
                    break
        if snapshot(args.export_dir, args.vectors_dir, library) != before:
            raise ValueError("Code or artifact SHA256 identity changed during verification")
        publish_live(args.live_dir, "completed", samples, len(roundtrips), duration)
    except Exception as error:
        if stream_started is not None:
            duration = time.perf_counter() - stream_started
        publish_live(args.live_dir, "failed", samples, len(roundtrips), duration, str(error))
        raise
    throughput = len(roundtrips) / duration
    return {
        "count_requested": args.count, "count_verified": len(roundtrips),
        "transport": "libftdi", "usb_serial": args.usb_serial,
        "usb_requested_baudrate": 460800, "baudrate": 115200, "clock_mhz": 50,
        "reset_verified": True, "elapsed_s": time.perf_counter() - started,
        "duration_s": duration, "duration_requested_s": args.duration, "throughput_hz": throughput,
        "acceptance_600s": duration >= 600 and throughput >= 100,
        "roundtrip_ms": summary(roundtrips), "compute_us": summary(compute),
        "export_manifest_sha256": before["export/manifest.json"],
        "vectors_manifest_sha256": before["vectors/manifest.json"],
        "checkpoint_sha256": artifact["checkpoint_sha256"], "provenance": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usb-serial", required=True, type=usb_serial_number)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--vectors-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=request_count)
    mode.add_argument("--duration", type=positive_duration)
    parser.add_argument("--libftdi")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live-dir", type=Path)
    args = parser.parse_args()
    if args.count is None and args.duration is None:
        args.count = 10000
    report = None
    try:
        report = verify(args)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(encoded)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if report is not None and args.live_dir is not None:
            latest_path = args.live_dir / "latest.json"
            if latest_path.is_file():
                latest = json.loads(latest_path.read_text())
                publish_live(args.live_dir, "failed", latest["samples"],
                             latest["count_verified"], latest["duration_s"], str(error))
        print(f"CfC verification failed: {error}", file=sys.stderr)
        return 1
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
