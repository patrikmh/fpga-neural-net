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


def recovery_testbench(exported):
    return f"""`timescale 1ns/1ps
module cfc_recovery_tb;
    localparam realtime BIT_NS = 1.0e9 / 115200.0;
    reg clk=0, rst=1, uart_rx=1;
    wire uart_tx;
    reg [7:0] request [0:127];
    reg [7:0] received;
    reg forbid_response=0;
    integer mode,n,frames,outfile,captured=0,bit_index;
    string reqpath,resppath;
    cfc_top #(.WEIGHTS_FILE({json.dumps(str(exported / 'weights.hex'))}),
              .TANH_FILE({json.dumps(str(exported / 'tanh.hex'))}),
              .SIGMOID_FILE({json.dumps(str(exported / 'sigmoid.hex'))}))
        dut(.clk(clk),.rst(rst),.uart_rx(uart_rx),.uart_tx(uart_tx));
    always #10 clk=~clk;
    task automatic send_byte(input reg [7:0] value);
        integer bit_number;
        begin
            uart_rx=0; #(BIT_NS);
            for(bit_number=0;bit_number<8;bit_number=bit_number+1) begin
                uart_rx=value[bit_number]; #(BIT_NS);
            end
            uart_rx=1; #(BIT_NS);
        end
    endtask
    initial begin
        wait(rst==0);
        forever begin
            @(negedge uart_tx);
            if(forbid_response) $fatal(1,"Rejected request unexpectedly produced UART response");
            #(BIT_NS/2.0);
            if(uart_tx!==0) $fatal(1,"Invalid start bit");
            for(bit_index=0;bit_index<8;bit_index=bit_index+1) begin
                #(BIT_NS); received[bit_index]=uart_tx;
            end
            #(BIT_NS);
            if(uart_tx!==1) $fatal(1,"Invalid stop bit");
            $fdisplay(outfile,"%02x",received);
            captured=captured+1;
        end
    end
    initial begin #30000000; $fatal(1,"CfC recovery response timeout after 30 ms"); end
    initial begin : stimulus
        integer index,start_index,frame_index,expected;
        if(!$value$plusargs("REQ=%s",reqpath) || !$value$plusargs("RESP=%s",resppath) ||
           !$value$plusargs("MODE=%d",mode) || !$value$plusargs("N=%d",n) ||
           !$value$plusargs("FRAMES=%d",frames)) $fatal(1,"Missing scenario arguments");
        $readmemh(reqpath,request,0,n-1);
        outfile=$fopen(resppath,"w");
        if(!outfile) $fatal(1,"Cannot open response file");
        repeat(40) @(posedge clk);
        @(negedge clk); rst=0;
        repeat(1024) @(posedge clk);
        @(negedge clk);
        start_index=0;
        if(mode==2) begin
            forbid_response=1;
            for(index=0;index<4;index=index+1) send_byte(request[index]);
            @(negedge clk); rst=1;
            repeat(40) @(posedge clk);
            @(negedge clk); rst=0;
            repeat(1024) @(posedge clk);
            @(negedge clk); forbid_response=0;
            start_index=4;
        end
        if(mode==3) begin send_byte(request[0]); start_index=1; end
        frame_index=0; expected=0;
        for(index=start_index;index<n;index=index+1) begin
            if(mode==1 && frame_index==1) forbid_response=1;
            send_byte(request[index]);
            if((index-start_index+1)%8==0) begin
                if(mode==1 && frame_index==1) begin
                    #(BIT_NS*40); forbid_response=0;
                end else begin
                    expected=expected+28;
                    wait(captured>=expected);
                    #(BIT_NS*2);
                end
                frame_index=frame_index+1;
            end
        end
        wait(captured>=frames*28);
        #(BIT_NS*40);
        if(captured!=frames*28) $fatal(1,"Unexpected extra response bytes");
        $fclose(outfile); $finish;
    end
endmodule
"""


# Behaviors: sequential steps plus protocol RESET; invalid CRC state isolation;
# leading-garbage resynchronization; global reset discarding a partial request.
class CfcUartRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        cls.exported = workspace / "exported"
        trained = workspace / "trained"
        commands = [
            ["model.train", "--output-dir", str(trained), "--epochs", "1", "--seed", "17",
             "--train-trajectories", "2", "--validation-trajectories", "1", "--length", "64"],
            ["model.export_integer", "--checkpoint", str(trained / "checkpoint.pt"),
             "--output-dir", str(cls.exported)],
        ]
        for command in commands:
            result = subprocess.run([PYTHON,"-m",*command],cwd=ROOT,capture_output=True,text=True,timeout=60)
            if result.returncode:
                raise AssertionError("Real UART fixture must build:\n"+result.stdout+result.stderr)

    def exchange(self, mode, payload, expected):
        from host.cfc_protocol import parse_response

        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            bench=workspace/"recovery.sv"
            bench.write_text(recovery_testbench(self.exported))
            simulation=workspace/"recovery.vvp"
            compiled=subprocess.run(
                [executable("iverilog"),"-g2012","-s","cfc_recovery_tb","-o",str(simulation),
                 "rtl/uart.sv","rtl/cfc_core.sv","rtl/cfc_top.sv",str(bench)],
                cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            request,response=workspace/"request.hex",workspace/"response.hex"
            request.write_text("".join(f"{byte:02x}\n" for byte in payload))
            simulated=subprocess.run(
                [executable("vvp"),str(simulation),f"+REQ={request}",f"+RESP={response}",
                 f"+MODE={mode}",f"+N={len(payload)}",f"+FRAMES={len(expected)}"],
                cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(simulated.returncode,0,simulated.stdout+simulated.stderr)
            raw=bytes.fromhex(response.read_text())
            self.assertEqual(len(raw),28*len(expected))
            for index,(sequence,y,hidden,is_reset) in enumerate(expected):
                with self.subTest(response=index):
                    seq_actual,y_actual,h_actual,cycles=parse_response(raw[index*28:(index+1)*28])
                    self.assertEqual((seq_actual,y_actual,h_actual),(sequence,y,tuple(hidden)))
                    if is_reset:
                        self.assertEqual(cycles,0)
                    else:
                        self.assertGreater(cycles,0)
                        self.assertLess(cycles,5000)

    def test_sequential_signed_steps_and_protocol_reset_return_exact_state(self):
        from host.cfc_protocol import build_request
        from model.integer import GoldenIntegerPredictor

        predictor=GoldenIntegerPredictor.from_export(self.exported)
        frames=[]
        expected=[]
        for sequence,x in enumerate((-32768,32767,0),1):
            hidden,y=predictor.step(x)
            frames.append(build_request(sequence,x))
            expected.append((sequence,y,hidden,False))
        frames.append(build_request(4,0,command=2))
        expected.append((4,0,[0]*8,True))
        predictor.reset()
        hidden,y=predictor.step(4096)
        frames.append(build_request(5,4096))
        expected.append((5,y,hidden,False))
        self.exchange(0,b"".join(frames),expected)

    def test_bad_crc_is_silent_and_does_not_change_recurrent_state(self):
        from host.cfc_protocol import build_request
        from model.integer import GoldenIntegerPredictor

        predictor=GoldenIntegerPredictor.from_export(self.exported)
        first_h,first_y=predictor.step(4096)
        next_h,next_y=predictor.step(-2048)
        bad=build_request(11,32767)
        bad=bad[:-1]+bytes([bad[-1]^1])
        self.exchange(1,build_request(10,4096)+bad+build_request(12,-2048),
                      [(10,first_y,first_h,False),(12,next_y,next_h,False)])

    def test_leading_garbage_is_ignored_before_a_valid_step(self):
        from host.cfc_protocol import build_request
        from model.integer import GoldenIntegerPredictor

        hidden,y=GoldenIntegerPredictor.from_export(self.exported).step(-4096)
        self.exchange(3,b"\x00"+build_request(21,-4096),[(21,y,hidden,False)])

    def test_global_reset_discards_partial_packet_before_fresh_step(self):
        from host.cfc_protocol import build_request
        from model.integer import GoldenIntegerPredictor

        hidden,y=GoldenIntegerPredictor.from_export(self.exported).step(2048)
        self.exchange(2,build_request(30,32767)[:4]+build_request(31,2048),[(31,y,hidden,False)])


if __name__ == "__main__":
    unittest.main()
