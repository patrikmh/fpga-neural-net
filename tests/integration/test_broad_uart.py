"""BROAD UART-to-USB verifier tracer using actual captured RTL replies."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

from integration.test_broad_train import prepared_fixture
from integration.test_probe_uart import executable


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_crc(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xffff if crc & 0x8000 else (crc << 1) & 0xffff
    return crc


def uart_bench(exported, banks, bad_crc=False):
    return f"""`timescale 1ns/1ps
module broad_uart_tb;
    localparam realtime BIT_NS=1.0e9/115200.0;
    reg clk=0,rst=1,uart_rx=1;
    wire uart_tx;
    reg [7:0] request [0:{95 if bad_crc else 71}];
    reg [7:0] received;
    integer replies,cycle_file,frame_index,byte_index,bit_index,captured=0;
    integer measured=0,measurement_count=0;
    reg was_busy=0;
    reg forbid_response=0;
    string request_path,response_path,cycle_path;
    broad_top #(.HIDDEN_SIZE(3),.WEIGHTS_DIR({json.dumps(str(banks))}),
                .TANH_FILE({json.dumps(str(exported/'tanh.hex'))}),
                .SIGMOID_FILE({json.dumps(str(exported/'sigmoid.hex'))}))
        dut(.clk(clk),.rst(rst),.uart_rx(uart_rx),.uart_tx(uart_tx));
    always #10 clk=~clk;
    always @(negedge clk) begin
        if(dut.core.busy===1) begin
            if(!was_busy) measured=1;
            else measured=measured+1;
            was_busy=1;
        end else if(was_busy) begin
            measured=measured+1;
            if(dut.core.done!==1) $fatal(1,"Core stopped without done");
            $fdisplay(cycle_file,"%0d",measured);
            measurement_count=measurement_count+1;
            was_busy=0;
        end
    end
    task automatic send_byte(input reg[7:0] value);
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
        for(byte_index=0;byte_index<72;byte_index=byte_index+1) begin
            @(negedge uart_tx);
            if(forbid_response) $fatal(1,"Bad CRC must produce no UART response");
            #(BIT_NS/2.0);
            if(uart_tx!==0) $fatal(1,"Invalid UART start");
            for(bit_index=0;bit_index<8;bit_index=bit_index+1) begin
                #(BIT_NS); received[bit_index]=uart_tx;
            end
            #(BIT_NS);
            if(uart_tx!==1) $fatal(1,"Invalid UART stop");
            $fdisplay(replies,"%02x",received); captured=captured+1;
        end
    end
    initial begin #40000000; $fatal(1,"BROAD UART response deadline expired"); end
    initial begin : stimulus
        integer index,completed_frames;
        if(!$value$plusargs("REQ=%s",request_path) || !$value$plusargs("RESP=%s",response_path) ||
           !$value$plusargs("CYCLES=%s",cycle_path)) $fatal(1,"Missing fixture paths");
        replies=$fopen(response_path,"w"); cycle_file=$fopen(cycle_path,"w");
        if(!replies || !cycle_file) $fatal(1,"Cannot open capture files");
        $readmemh(request_path,request);
        repeat(40) @(posedge clk);
        @(negedge clk); rst=0;
        repeat(1024) @(posedge clk);
        @(negedge clk);
        completed_frames=0;
        for(frame_index=0;frame_index<{4 if bad_crc else 3};frame_index=frame_index+1) begin
            if({1 if bad_crc else 0} && frame_index==1) forbid_response=1;
            for(index=0;index<24;index=index+1) send_byte(request[frame_index*24+index]);
            if(forbid_response) begin
                #(BIT_NS*40); forbid_response=0;
            end else begin
                completed_frames=completed_frames+1;
                wait(captured>=completed_frames*24);
                #(BIT_NS*2);
            end
        end
        if(measurement_count!=2) $fatal(1,"Expected two directly measured core transactions");
        $fclose(replies); $fclose(cycle_file); $finish;
    end
endmodule
"""


class BroadUartVerifierTracerTests(unittest.TestCase):
    def exchange_pipeline(self, bad_crc=False, native_fault=None):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            prepared,trained,exported,banks,vectors=(workspace/name for name in ("prepared","trained","exported","banks","vectors"))
            prepared_fixture(prepared)
            commands=[
                ["model.broad_train","--prepared-dir",str(prepared),"--output-dir",str(trained),"--hidden-size","3",
                 "--epochs","1","--updates-per-epoch","2","--batch-size","2","--window","16",
                 "--warmup","2","--validation-every","1","--seed","17"],
                ["model.broad_export_integer","--checkpoint",str(trained/"checkpoint.pt"),"--output-dir",str(exported)],
            ]
            for command in commands:
                result=subprocess.run([PYTHON,"-m",*command],cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            from model.broad_integer import GoldenBroadPredictor

            predictor=GoldenBroadPredictor.from_export(exported)
            artifact=json.loads((exported/"manifest.json").read_text())
            before={path.name:sha256(path) for path in exported.iterdir() if path.is_file()}
            banks.mkdir(); vectors.mkdir()
            words=(exported/"weights.hex").read_text().split()
            for number,offset in enumerate(range(0,len(words),1024)):
                bank=words[offset:offset+1024]
                bank += ["0000"]*(1024-len(bank))
                (banks/f"weights_{number:03d}.hex").write_text("\n".join(bank)+"\n")
            inputs=[[-8192,-4096,-1,0,1,4096,8192,32767,-32768],
                    [2048,-2048,4096,-4096,8191,-8192,17,-17,0]]
            expected=[predictor.step(values) for values in inputs]
            rows=["".join(f"{value&65535:04x}" for value in values+hidden+raw)
                  for values,(hidden,raw) in zip(inputs,expected)]
            vector_file=vectors/"vectors.hex"
            vector_file.write_text("\n".join(rows)+"\n")
            vector_manifest={"count":2,"hidden_size":3,"initial_hidden":[0]*3,"numeric":artifact["numeric"],
                             "export_manifest_sha256":sha256(exported/"manifest.json"),
                             "checkpoint_sha256":artifact["checkpoint_sha256"],"vectors_sha256":sha256(vector_file)}
            (vectors/"manifest.json").write_text(json.dumps(vector_manifest))
            testbench=workspace/"broad_uart_tb.sv"
            testbench.write_text(uart_bench(exported,banks,bad_crc=bad_crc))
            simulation=workspace/"uart.vvp"
            compiled=subprocess.run([executable("iverilog"),"-g2012","-s","broad_uart_tb","-o",str(simulation),
                                     "rtl/uart.sv","rtl/broad_core.sv","rtl/broad_top.sv",str(testbench)],
                                    cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,"Actual BROAD UART RTL must compile:\n"+compiled.stdout+compiled.stderr)
            from host.broad_protocol import build_request,parse_response,crc16

            self.assertEqual(crc16(b"123456789"),0x29b1)
            frames=[build_request(65535,[0]*9,command=2)]+[build_request(index,values) for index,values in enumerate(inputs)]
            for index,request in enumerate(frames):
                sequence=65535 if index==0 else index-1
                values=[0]*9 if index==0 else inputs[index-1]
                payload=struct.pack("<BBH9h",0xa6,2 if index==0 else 1,sequence,*values)
                self.assertEqual(request,payload+struct.pack("<H",reference_crc(payload)))
            request_hex,response_hex,cycle_file=workspace/"request.hex",workspace/"response.hex",workspace/"cycles.txt"
            requests=b"".join(frames)
            wire_frames=list(frames)
            if bad_crc:
                invalid=build_request(99,[32767]*9)
                wire_frames.insert(1,invalid[:-1]+bytes([invalid[-1]^1]))
            request_hex.write_text("".join(f"{byte:02x}\n" for byte in b"".join(wire_frames)))
            simulated=subprocess.run([executable("vvp"),str(simulation),f"+REQ={request_hex}",
                                      f"+RESP={response_hex}",f"+CYCLES={cycle_file}"],
                                     cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(simulated.returncode,0,simulated.stdout+simulated.stderr)
            captured=bytes.fromhex(response_hex.read_text())
            self.assertEqual(len(captured),72)
            measured=[int(value) for value in cycle_file.read_text().split()]
            self.assertEqual(len(measured),2)
            self.assertEqual(measured[0],measured[1])
            self.assertGreater(measured[0],0)
            for index in range(3):
                reply=captured[index*24:(index+1)*24]
                self.assertEqual(reference_crc(reply[:-2]),int.from_bytes(reply[-2:],"little"))
                self.assertEqual(reply[:2],b"\xa6\x00")
                decoded=parse_response(reply,hidden_size=3)
                wanted=(65535,(0,)*3,(0,)*4,0) if index==0 else (index-1,tuple(expected[index-1][0]),tuple(expected[index-1][1]),measured[index-1])
                self.assertEqual(decoded,wanted)
            # Reuse the external device ABI fixture with this protocol's lengths/opcode offset.
            c_source=(ROOT/"tests/integration/fixtures/fake_cfc_ftdi.c").read_text()
            c_source=re.sub(r"\b28\b","24",c_source)
            for old,new in (("expected[8]","expected[24]"),("size != 8","size != 24"),
                            ("fread(expected, 1, 8, requests) != 8","fread(expected, 1, 24, requests) != 24"),
                            ("memcmp(expected, data, 8)","memcmp(expected, data, 24)"),("data[4]","data[1]")):
                c_source=c_source.replace(old,new)
            c_path,library=workspace/"fake_broad.c",workspace/"fake_broad.dylib"
            c_path.write_text(c_source)
            compiled=subprocess.run([shutil.which("clang") or "clang","-shared","-fPIC",str(c_path),"-o",str(library)],
                                    cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            request_bin,reply_bin,trace,output=(workspace/name for name in ("requests.bin","replies.bin","trace.txt","report.json"))
            request_bin.write_bytes(requests)
            native_replies=captured
            if native_fault=="sequence":
                changed=bytearray(captured[24:48])
                changed[2]^=1
                changed[-2:]=struct.pack("<H",reference_crc(changed[:-2]))
                native_replies=captured[:24]+bytes(changed)+captured[48:]
            reply_bin.write_bytes(native_replies)
            import os
            verified=subprocess.run([PYTHON,"-m","host.verify_broad","--usb-serial","2023112818",
                                     "--export-dir",str(exported),"--vectors-dir",str(vectors),"--count","2",
                                     "--expected-cycles",str(measured[0]),"--libftdi",str(library),"--output",str(output)],
                                    cwd=ROOT,env={**os.environ,"FAKE_CFC_REQUESTS":str(request_bin),
                                                  "FAKE_CFC_REPLIES":str(reply_bin),"FAKE_CFC_TRACE":str(trace),
                                                  "FAKE_CFC_FAULT":"short" if native_fault=="short" else ""},
                                    capture_output=True,text=True,timeout=10)
            if native_fault is not None:
                self.assertNotEqual(verified.returncode,0,verified.stdout+verified.stderr)
                self.assertEqual(verified.stdout.strip(),"")
                self.assertFalse(output.exists(),"Rejected exchange must not publish a successful report")
                self.assertRegex(verified.stderr.lower(),"sequence" if native_fault=="sequence" else r"(timeout|incomplete|response frame)")
                self.assertEqual(trace.read_text().splitlines()[-2:],["close","free"])
                return
            self.assertEqual(verified.returncode,0,"Real BROAD USB verifier must consume captured RTL replies:\n"+verified.stdout+verified.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(json.loads(verified.stdout),report)
            self.assertEqual(report["count_requested"],2)
            self.assertEqual(report["count_verified"],2)
            self.assertEqual(report["hidden_size"],3)
            self.assertTrue(report["reset_verified"])
            self.assertEqual(report["expected_cycles"],measured[0])
            for name in ("min","mean","p95","max"):
                self.assertAlmostEqual(report["compute_us"][name],measured[0]/50)
            self.assertEqual(report["export_manifest_sha256"],sha256(exported/"manifest.json"))
            self.assertEqual(report["vectors_manifest_sha256"],sha256(vectors/"manifest.json"))
            events=trace.read_text().splitlines()
            self.assertEqual(events.count("request RESET"),1)
            self.assertEqual(events.count("request STEP"),2)
            self.assertEqual(events[-2:],["close","free"])
            self.assertEqual({path.name:sha256(path) for path in exported.iterdir() if path.is_file()},before)

    def test_host_verifier_accepts_actual_crc_uart_core_replies_with_all_states(self):
        self.exchange_pipeline()

    def test_bad_crc_step_is_silent_and_does_not_advance_recurrent_state(self):
        self.exchange_pipeline(bad_crc=True)

    def test_native_wrong_sequence_or_incomplete_reply_cannot_publish_success(self):
        for fault in ("sequence","short"):
            with self.subTest(fault=fault):
                self.exchange_pipeline(native_fault=fault)


if __name__=="__main__":
    unittest.main()
