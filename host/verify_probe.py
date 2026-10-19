"""Verify a named UART probe without programming or discovering devices.

Sequences 0..2 use fixed signed boundary vectors. Subsequent sequences use
a=((seq*4051+17389)&65535)-32768, b=((seq*25173+13849)&65535)-32768,
and address=seq%256. The odd multipliers traverse the full signed 16-bit range;
the default 1000 requests exercise all 256 ROM addresses.
"""

import argparse
import json
import math
import statistics
import sys
import time

import serial

from host.protocol import build_request, parse_response
from host.ftdi import Ftdi


def request_count(value):
    count = int(value)
    if not 1 <= count <= 65536:
        raise argparse.ArgumentTypeError("count must be between 1 and 65536")
    return count


def usb_serial_number(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("USB serial number must not be empty")
    return value


def vector(sequence):
    initial = ((-32768, 32767, 0), (32767, -32768, 255), (0, 32767, 1))
    if sequence < len(initial):
        return initial[sequence]
    return (
        ((sequence * 4051 + 17389) & 65535) - 32768,
        ((sequence * 25173 + 13849) & 65535) - 32768,
        sequence % 256,
    )


def verify(port, count, usb_serial=None, libftdi=None):
    timings = []
    started = time.perf_counter()
    connection = Ftdi(usb_serial, libftdi) if usb_serial else serial.Serial(
        port=port, baudrate=115200, bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        timeout=1.0, write_timeout=1.0,
    )
    with connection:
        for sequence in range(count):
            a, b, address = vector(sequence)
            request = build_request(sequence, a, b, address)
            roundtrip_started = time.perf_counter()
            if connection.write(request) != len(request):
                raise ValueError(f"Incomplete request write at sequence {sequence}")
            response = connection.read(9)
            if len(response) != 9:
                raise ValueError(f"Response timeout at sequence {sequence}: received {len(response)}/9 bytes")
            actual_sequence, result = parse_response(response)
            expected = a * b + ((address * 4051 + 17389) & 65535) - 32768
            if actual_sequence != sequence:
                raise ValueError(f"Sequence mismatch: expected {sequence}, received {actual_sequence}")
            if result != expected:
                raise ValueError(f"Result mismatch at sequence {sequence}: expected {expected}, received {result}")
            timings.append((time.perf_counter() - roundtrip_started) * 1000)
    ordered = sorted(timings)
    report = {
        "count_requested": count, "count_verified": len(timings),
        "baudrate": 115200, "port": port,
        "elapsed_s": time.perf_counter() - started,
        "roundtrip_ms": {
            "min": ordered[0], "mean": statistics.mean(timings),
            "p95": ordered[math.ceil(count * 0.95) - 1], "max": ordered[-1],
        },
    }
    if usb_serial:
        report.update(transport="libftdi", usb_serial=usb_serial, usb_requested_baudrate=460800)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--port")
    transport.add_argument("--usb-serial", type=usb_serial_number)
    parser.add_argument("--libftdi")
    parser.add_argument("--count", type=request_count, default=1000)
    args = parser.parse_args()
    try:
        report = verify(args.port, args.count, args.usb_serial, args.libftdi)
    except (serial.SerialException, OSError, ValueError) as error:
        print(f"Probe verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
