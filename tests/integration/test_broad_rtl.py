"""Tiny real BROAD core tracer against a checked integer export."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_broad_train import prepared_fixture
from integration.test_probe_uart import executable


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def bench(banks, exported, vectors):
    return f"""`timescale 1ns/1ps
module broad_core_tb;
    reg clk=0, reset=1, start=0;
    reg [143:0] input_values=0;
    wire [47:0] hidden_values;
    wire [63:0] output_values;
    wire busy,done;
    wire [31:0] cycles;
    reg [255:0] vectors [0:3];
    integer step,lane,observed,first_cycles=0;
    broad_core #(.HIDDEN_SIZE(3),.WEIGHTS_DIR({json.dumps(str(banks))}),
                 .TANH_FILE({json.dumps(str(exported/'tanh.hex'))}),
                 .SIGMOID_FILE({json.dumps(str(exported/'sigmoid.hex'))}))
        dut(.clk(clk),.reset(reset),.start(start),.input_values(input_values),
            .hidden_values(hidden_values),.output_values(output_values),
            .busy(busy),.done(done),.cycles(cycles));
    always #10 clk=~clk;
    task automatic cold_reset;
        begin
            @(negedge clk); reset=1; start=0;
            repeat(5) @(negedge clk);
            if(busy!==0 || done!==0 || hidden_values!==0 || output_values!==0 || cycles!==0)
                $fatal(1,"Reset must clear externally visible core state");
            reset=0;
            repeat(5) @(negedge clk);
        end
    endtask
    initial begin #20000000; $fatal(1,"BROAD core total simulation watchdog expired"); end
    initial begin
        $readmemh({json.dumps(str(vectors))},vectors);
        cold_reset();
        for(step=0;step<4;step=step+1) begin
            if(step==3) cold_reset();
            @(negedge clk);
            if(busy!==0 || done!==0) $fatal(1,"Core not idle before step %0d",step);
            for(lane=0;lane<9;lane=lane+1)
                input_values[lane*16 +:16]=vectors[step][(15-lane)*16 +:16];
            start=1;
            @(negedge clk); observed=1; start=0;
            if(busy!==1) $fatal(1,"Start was not accepted at step %0d",step);
            while(done!==1 && observed<100000) begin
                if(busy!==1) $fatal(1,"Busy ended before done at step %0d",step);
                @(negedge clk); observed=observed+1;
            end
            if(done!==1 || observed>=100000) $fatal(1,"Per-step deadline at %0d",step);
            if(busy!==0 || cycles!==observed)
                $fatal(1,"Observed/report cycle or completion mismatch at %0d: %0d/%0d",step,observed,cycles);
            if(step==0) first_cycles=observed;
            else if(observed!=first_cycles) $fatal(1,"Data-dependent latency at step %0d",step);
            for(lane=0;lane<3;lane=lane+1)
                if(hidden_values[lane*16 +:16]!==vectors[step][(6-lane)*16 +:16])
                    $fatal(1,"Hidden mismatch step %0d lane %0d",step,lane);
            for(lane=0;lane<4;lane=lane+1)
                if(output_values[lane*16 +:16]!==vectors[step][(3-lane)*16 +:16])
                    $fatal(1,"Raw quaternion mismatch step %0d lane %0d",step,lane);
            @(negedge clk);
            if(done!==0 || busy!==0) $fatal(1,"Completion must pulse for one clock");
        end
        $display("PASS: four BROAD steps, recurrence, cold repeat, and actual cycle counts matched (%0d cycles)",first_cycles);
        $finish;
    end
endmodule
"""


class BroadRtlTracerTests(unittest.TestCase):
    def test_clocked_broad_core_matches_recurrent_hidden_raw_outputs_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, trained, exported = (workspace/name for name in ("prepared","trained","exported"))
            prepared_fixture(prepared)
            commands = [
                ["model.broad_train","--prepared-dir",str(prepared),"--output-dir",str(trained),
                 "--hidden-size","3","--epochs","1","--updates-per-epoch","2","--batch-size","2",
                 "--window","16","--warmup","2","--validation-every","1","--seed","17"],
                ["model.broad_export_integer","--checkpoint",str(trained/"checkpoint.pt"),"--output-dir",str(exported)],
            ]
            for command in commands:
                result=subprocess.run([PYTHON,"-m",*command],cwd=ROOT,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            from model.broad_integer import GoldenBroadPredictor

            predictor=GoldenBroadPredictor.from_export(exported)
            before=hashes(exported)
            manifest=json.loads((exported/"manifest.json").read_text())
            for filename in ("weights.hex","tanh.hex","sigmoid.hex"):
                self.assertEqual(manifest["files"][filename],before[filename])
            banks=workspace/"banks"
            banks.mkdir()
            words=(exported/"weights.hex").read_text().split()
            for number,offset in enumerate(range(0,len(words),1024)):
                bank=words[offset:offset+1024]
                bank+= ["0000"]*(1024-len(bank))
                (banks/f"weights_{number:03d}.hex").write_text("\n".join(bank)+"\n")
            samples=[[-8192,-4096,-1,0,1,4096,8192,32767,-32768],
                     [2048,-2048,4096,-4096,8191,-8192,17,-17,0],
                     [-1,-2,-3,-4,-5,-6,-7,-8,-9]]
            samples.append(samples[0].copy())
            rows=[]
            results=[]
            for index,inputs in enumerate(samples):
                if index==3:
                    predictor.reset()
                hidden,raw=predictor.step(inputs)
                results.append((hidden,raw))
                rows.append("".join(f"{value&65535:04x}" for value in inputs+hidden+raw))
            self.assertEqual(results[3],results[0])
            self.assertTrue(any(value!=0 for value in results[0][0]),"Fixture must exercise nonzero recurrent state")
            vectors=workspace/"vectors.hex"
            vectors.write_text("\n".join(rows)+"\n")
            testbench=workspace/"broad_core_tb.sv"
            testbench.write_text(bench(banks,exported,vectors))
            simulation=workspace/"broad_core.vvp"
            compiled=subprocess.run([executable("iverilog"),"-g2012","-s","broad_core_tb","-o",str(simulation),
                                     "rtl/broad_core.sv",str(testbench)],
                                    cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,"Real BROAD core RTL must compile:\n"+compiled.stdout+compiled.stderr)
            simulated=subprocess.run([executable("vvp"),str(simulation)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(simulated.returncode,0,simulated.stdout+simulated.stderr)
            self.assertIn("PASS: four BROAD steps",simulated.stdout)
            self.assertEqual(hashes(exported),before)


if __name__=="__main__":
    unittest.main()
