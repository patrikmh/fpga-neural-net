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
    def test_production_blend_value_matches_valid_activation_domain_reference(self):
        import random

        endpoints=(-8192,-8191,-1,0,1,8191,8192)
        cases={(f1,f2,gate) for f1 in endpoints for f2 in endpoints for gate in (0,1,4096,8191,8192)}
        cases.update(((0,1,4096),(0,-1,4096),(1,-2,4096),(-1,2,4096),
                      (924,-128,3793),(1332,-1580,3176),(2567,-64,593),
                      (702,-3710,520),(7719,-702,1279),(7923,797,444)))
        rng=random.Random(20261021)
        cases.update((rng.randrange(-8192,8193),rng.randrange(-8192,8193),rng.randrange(8193)) for _ in range(2048))
        rows=[]
        for f1,f2,gate in sorted(cases):
            product=(f2-f1)*gate
            rounded=(abs(product)+4096)//8192
            if product<0:
                rounded=-rounded
            self.assertTrue(-32768<=rounded<=32767)
            total=f1+rounded
            self.assertTrue(-(1<<16)<=total<(1<<16))
            expected=min(32767,max(-32768,total))
            rows.append("".join(f"{value&65535:04x}" for value in (f1,f2-f1,gate,expected)))
        core=ROOT/"rtl/broad_core.sv"
        before=hashlib.sha256(core.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            banks=workspace/"banks"
            banks.mkdir()
            (banks/"weights_000.hex").write_text("0000\n"*1024)
            table=workspace/"activation.hex"
            table.write_text("0000\n"*4096)
            vectors=workspace/"blend.hex"
            vectors.write_text("\n".join(rows)+"\n")
            testbench=workspace/"blend_tb.sv"
            testbench.write_text(f"""`timescale 1ns/1ps
module broad_blend_tb;
    reg [63:0] vectors [0:{len(rows)-1}];
    integer index;
    broad_core #(.HIDDEN_SIZE(1),.WEIGHTS_DIR({json.dumps(str(banks))}),
                 .TANH_FILE({json.dumps(str(table))}),.SIGMOID_FILE({json.dumps(str(table))}))
        dut(.clk(1'b0),.reset(1'b0),.start(1'b0),.input_values(144'b0),
            .hidden_values(),.output_values(),.busy(),.done(),.cycles());
    initial begin
        $readmemh({json.dumps(str(vectors))},vectors);
        for(index=0;index<{len(rows)};index=index+1) begin
            dut.f1=$signed(vectors[index][63:48]);
            dut.mul_a=$signed(vectors[index][47:32]);
            dut.mul_b=$signed(vectors[index][31:16]);
            #1;
            if(dut.blend_value!==vectors[index][15:0])
                $fatal(1,"Production blend_value mismatch row %0d f1=%0d delta=%0d gate=%0d got=%0d expected=%0d",
                       index,dut.f1,dut.mul_a,dut.mul_b,dut.blend_value,$signed(vectors[index][15:0]));
        end
        $display("PASS: {len(rows)} valid activation triples matched production blend_value");
        $finish;
    end
endmodule
""")
            simulation=workspace/"blend.vvp"
            compiled=subprocess.run([executable("iverilog"),"-g2012","-s","broad_blend_tb","-o",str(simulation),
                                     str(core),str(testbench)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            simulated=subprocess.run([executable("vvp"),str(simulation)],cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(simulated.returncode,0,simulated.stdout+simulated.stderr)
            self.assertIn(f"PASS: {len(rows)} valid activation triples",simulated.stdout)
        self.assertEqual(hashlib.sha256(core.read_bytes()).hexdigest(),before)

    def test_production_round_shift_matches_full_signed48_reference_at_boundaries_and_random_values(self):
        import random

        minimum,maximum=-(1<<47),(1<<47)-1
        values={minimum,minimum+1,minimum+2,maximum,maximum-1,maximum-2,0,-1,1,
                -3990236,-9248512,-1560183,-2294240,-10770459,-3163944,-3701173,-3634404,-3508388}
        for bits in (13,18):
            half=1<<(bits-1)
            largest=maximum>>bits
            for quotient in (0,1,2,3,2047,2048,2049,largest-1,largest):
                base=quotient<<bits
                for sign in (-1,1):
                    for candidate in (base,base-1,base+1,base+half-1,base+half,base+half+1):
                        value=sign*candidate
                        if minimum<=value<=maximum:
                            values.add(value)
            for index in (-2049,-2048,-2047,2046,2047,2048):
                for delta in (-half-1,-half,-half+1,half-1,half,half+1):
                    values.add((index<<18)+delta)
        rng=random.Random(20261020)
        values.update(rng.randrange(minimum,maximum+1) for _ in range(2048))
        values=sorted(values)
        def reference(value,bits):
            rounded=(abs(value)+(1<<(bits-1)))//(1<<bits)
            return -rounded if value<0 else rounded
        mask=(1<<48)-1
        rows=[]
        for value in values:
            expected13,expected18=reference(value,13),reference(value,18)
            index=min(4095,max(0,expected18+2048))
            rows.append(f"{value&mask:012x}{expected13&mask:012x}{expected18&mask:012x}{index:03x}")
        core=ROOT/"rtl/broad_core.sv"
        before=hashlib.sha256(core.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            banks=workspace/"banks"
            banks.mkdir()
            (banks/"weights_000.hex").write_text("0000\n"*1024)
            table=workspace/"activation.hex"
            table.write_text("0000\n"*4096)
            vectors=workspace/"rounding.hex"
            vectors.write_text("\n".join(rows)+"\n")
            testbench=workspace/"rounding_tb.sv"
            testbench.write_text(f"""`timescale 1ns/1ps
module broad_rounding_tb;
    reg [155:0] vectors [0:{len(values)-1}];
    reg signed [47:0] value,actual;
    reg [11:0] actual_index;
    integer index;
    broad_core #(.HIDDEN_SIZE(1),.WEIGHTS_DIR({json.dumps(str(banks))}),
                 .TANH_FILE({json.dumps(str(table))}),.SIGMOID_FILE({json.dumps(str(table))}))
        dut(.clk(1'b0),.reset(1'b1),.start(1'b0),.input_values(144'b0),
            .hidden_values(),.output_values(),.busy(),.done(),.cycles());
    initial begin
        $readmemh({json.dumps(str(vectors))},vectors);
        for(index=0;index<{len(values)};index=index+1) begin
            value=$signed(vectors[index][155:108]);
            actual=dut.round_shift(value,13);
            if(actual!==vectors[index][107:60])
                $fatal(1,"Full signed48 round_shift13 mismatch value=%0d got=%0d expected=%0d",value,actual,$signed(vectors[index][107:60]));
            actual=dut.round_shift(value,18);
            if(actual!==vectors[index][59:12])
                $fatal(1,"Full signed48 round_shift18 mismatch value=%0d got=%0d expected=%0d",value,actual,$signed(vectors[index][59:12]));
            actual_index=dut.lookup_index(value);
            if(actual_index!==vectors[index][11:0])
                $fatal(1,"Activation clamp mismatch value=%0d",value);
            #1;
        end
        $display("PASS: {len(values)} signed48 values, production round_shift13/18 and activation clamping");
        $finish;
    end
endmodule
""")
            simulation=workspace/"rounding.vvp"
            compiled=subprocess.run([executable("iverilog"),"-g2012","-s","broad_rounding_tb","-o",str(simulation),
                                     str(core),str(testbench)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(compiled.returncode,0,compiled.stdout+compiled.stderr)
            simulated=subprocess.run([executable("vvp"),str(simulation)],cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(simulated.returncode,0,simulated.stdout+simulated.stderr)
            self.assertIn(f"PASS: {len(values)} signed48 values",simulated.stdout)
        self.assertEqual(hashlib.sha256(core.read_bytes()).hexdigest(),before)

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
