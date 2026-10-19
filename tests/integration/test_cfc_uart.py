"""First full CfC path from host protocol through UART pins and real RTL."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_probe_uart import executable


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def testbench(exported):
    return f"""`timescale 1ns/1ps
module cfc_uart_tb;
    localparam realtime BIT_NS = 1.0e9 / 115200.0;
    reg clk = 0;
    reg rst = 1;
    reg uart_rx = 1;
    wire uart_tx;
    reg [7:0] request [0:7];
    reg [7:0] received;
    string request_path, response_path;
    integer response_file, byte_index, bit_index;
    cfc_top #(
        .WEIGHTS_FILE({json.dumps(str(exported / 'weights.hex'))}),
        .TANH_FILE({json.dumps(str(exported / 'tanh.hex'))}),
        .SIGMOID_FILE({json.dumps(str(exported / 'sigmoid.hex'))})
    ) dut(.clk(clk), .rst(rst), .uart_rx(uart_rx), .uart_tx(uart_tx));
    always #10 clk = ~clk;

    task automatic send_byte(input reg [7:0] value);
        integer bit_number;
        begin
            uart_rx = 0;
            #(BIT_NS);
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                uart_rx = value[bit_number];
                #(BIT_NS);
            end
            uart_rx = 1;
            #(BIT_NS);
        end
    endtask

    initial begin
        #20000000;
        $fatal(1, "Complete CfC UART response not received within 20 ms");
    end

    initial begin
        if (!$value$plusargs("REQ=%s", request_path) ||
            !$value$plusargs("RESP=%s", response_path))
            $fatal(1, "Missing request/response fixture arguments");
        $readmemh(request_path, request);
        response_file = $fopen(response_path, "w");
        if (!response_file) $fatal(1, "Cannot open response destination");
        repeat (40) @(posedge clk);
        @(negedge clk);
        rst = 0;
        repeat (1024) @(posedge clk);
        @(negedge clk);
        fork
            begin : transmit_request
                integer index;
                for (index = 0; index < 8; index = index + 1)
                    send_byte(request[index]);
            end
            begin : capture_response
                for (byte_index = 0; byte_index < 28; byte_index = byte_index + 1) begin
                    @(negedge uart_tx);
                    #(BIT_NS / 2.0);
                    if (uart_tx !== 0) $fatal(1, "Invalid UART start bit");
                    for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                        #(BIT_NS);
                        received[bit_index] = uart_tx;
                    end
                    #(BIT_NS);
                    if (uart_tx !== 1) $fatal(1, "Invalid UART stop bit");
                    $fdisplay(response_file, "%02x", received);
                end
            end
        join
        $fclose(response_file);
        $finish;
    end
endmodule
"""


class CfcUartTracerTests(unittest.TestCase):
    def test_host_uart_step_returns_complete_cfc_state_and_bounded_cycle_count(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            trained, exported = workspace / "trained", workspace / "exported"
            commands = [
                ["model.train", "--output-dir", str(trained), "--epochs", "1", "--seed", "17",
                 "--train-trajectories", "2", "--validation-trajectories", "1", "--length", "64"],
                ["model.export_integer", "--checkpoint", str(trained / "checkpoint.pt"),
                 "--output-dir", str(exported)],
            ]
            for command in commands:
                result = subprocess.run([PYTHON, "-m", *command], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            bench = workspace / "cfc_uart_tb.sv"
            bench.write_text(testbench(exported))
            simulation = workspace / "cfc.vvp"
            compiled = subprocess.run(
                [executable("iverilog"), "-g2012", "-s", "cfc_uart_tb", "-o", str(simulation),
                 "rtl/uart.sv", "rtl/cfc_core.sv", "rtl/cfc_top.sv", str(bench)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0,
                             "Real CfC RTL must compile:\n" + compiled.stdout + compiled.stderr)

            from host.cfc_protocol import build_request, parse_response
            from model.integer import GoldenIntegerPredictor

            sequence, x = 0x1234, 4096
            request = build_request(sequence, x, command=1)
            self.assertEqual(len(request), 8)
            request_path, response_path = workspace / "request.hex", workspace / "response.hex"
            request_path.write_text("".join(f"{byte:02x}\n" for byte in request))
            simulated = subprocess.run(
                [executable("vvp"), str(simulation), f"+REQ={request_path}", f"+RESP={response_path}"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(simulated.returncode, 0,
                             "Complete CfC UART exchange must finish:\n" + simulated.stdout + simulated.stderr)
            response = bytes.fromhex(response_path.read_text())
            self.assertEqual(len(response), 28)
            actual_sequence, actual_y, actual_hidden, cycles = parse_response(response)
            expected_hidden, expected_y = GoldenIntegerPredictor.from_export(exported).step(x)
            self.assertEqual((actual_sequence, actual_y, actual_hidden),
                             (sequence, expected_y, tuple(expected_hidden)))
            self.assertGreater(cycles, 0)
            self.assertLess(cycles, 5000)


if __name__ == "__main__":
    unittest.main()
