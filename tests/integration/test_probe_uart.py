"""Tracer from the host protocol through physical UART pins and back."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LOCAL_BIN = ROOT / ".tools/tang-toolchain/oss-cad-suite/bin"


def executable(name):
    return shutil.which(name) or str(LOCAL_BIN / name)


class ProbeUartTracerTests(unittest.TestCase):
    def test_host_request_returns_sequence_and_signed_multiply_plus_rom_over_uart(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            simulation = workspace / "probe.vvp"
            compiled = subprocess.run(
                [
                    executable("iverilog"), "-g2012", "-s", "probe_uart_tb",
                    "-o", str(simulation), "rtl/uart.sv", "rtl/probe_top.sv",
                    "tests/integration/probe_uart_tb.sv",
                ],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(
                compiled.returncode, 0,
                "UART probe RTL must compile:\n" + compiled.stdout + compiled.stderr,
            )

            # Import only after compilation so absent RTL is the first RED boundary.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from host.protocol import build_request, parse_response

            sequence, a, b, address = 0x1234, -123, 17, 17
            request = build_request(sequence, a, b, address)
            request_file = workspace / "request.hex"
            request_file.write_text("".join(f"{byte:02x}\n" for byte in request))
            response_file = workspace / "response.hex"
            simulated = subprocess.run(
                [executable("vvp"), str(simulation),
                 f"+REQ={request_file}", f"+RESP={response_file}"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(
                simulated.returncode, 0,
                "UART probe must complete a valid pin-level exchange:\n"
                + simulated.stdout + simulated.stderr,
            )
            response = bytes.fromhex(response_file.read_text())
            rom_value = ((address * 4051 + 17389) & 65535) - 32768
            self.assertEqual(parse_response(response), (sequence, a * b + rom_value))


if __name__ == "__main__":
    unittest.main()
