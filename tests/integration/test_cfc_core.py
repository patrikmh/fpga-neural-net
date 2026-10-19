"""Continuous real-core simulation against a disposable 10,000-step export."""

# Behavior: the full core preserves every reference state/output and reports
# its actual accepted-start-to-done latency throughout one continuous sequence.

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_probe_uart import executable


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def core_testbench(exported, vectors):
    return f"""`timescale 1ns/1ps
module cfc_core_long_tb;
    reg clk = 0;
    reg rst = 1;
    reg start = 0;
    reg signed [15:0] sample = 0;
    wire busy, done;
    wire signed [15:0] y;
    wire [127:0] hidden;
    wire [31:0] cycles;
    reg [159:0] vectors [0:9999];
    integer index, neuron, observed;
    cfc_core #(
        .WEIGHTS_FILE({json.dumps(str(exported / 'weights.hex'))}),
        .TANH_FILE({json.dumps(str(exported / 'tanh.hex'))}),
        .SIGMOID_FILE({json.dumps(str(exported / 'sigmoid.hex'))})
    ) dut(.clk(clk), .rst(rst), .start(start), .sample(sample),
          .busy(busy), .done(done), .y(y), .hidden(hidden), .cycles(cycles));
    always #10 clk = ~clk;

    initial begin
        #1100000000;
        $fatal(1, "Continuous CfC simulation exceeded total watchdog");
    end

    initial begin
        $readmemh({json.dumps(str(vectors / 'vectors.hex'))}, vectors);
        repeat (40) @(posedge clk);
        @(negedge clk);
        rst = 0;
        for (index = 0; index < 10000; index = index + 1) begin
            @(negedge clk);
            if (busy !== 0) $fatal(1, "Core not idle before step %0d", index);
            sample = vectors[index][159:144];
            start = 1;
            @(negedge clk);
            // Exactly one rising edge has accepted start since it was asserted.
            observed = 1;
            start = 0;
            if (busy !== 1) $fatal(1, "Core did not accept start at step %0d", index);
            while (done !== 1 && observed < 5000) begin
                @(negedge clk);
                observed = observed + 1;
            end
            if (done !== 1 || observed >= 5000)
                $fatal(1, "Core exceeded per-step deadline at step %0d", index);
            if (cycles !== observed)
                $fatal(1, "Reported cycles %0d differ from observed %0d at step %0d", cycles, observed, index);
            if (busy !== 0) $fatal(1, "Core busy at completion of step %0d", index);
            if (y !== vectors[index][15:0])
                $fatal(1, "Output differs at step %0d: got %0d expected %0d", index, y, $signed(vectors[index][15:0]));
            for (neuron = 0; neuron < 8; neuron = neuron + 1)
                if (hidden[neuron*16 +: 16] !== vectors[index][(8-neuron)*16 +: 16])
                    $fatal(1, "Hidden state differs at step %0d neuron %0d", index, neuron);
        end
        $display("PASS: 10000 continuous steps, all 8 hidden states/output and observed cycle counts matched");
        $finish;
    end
endmodule
"""


class CfcCoreContinuousTests(unittest.TestCase):
    def test_core_matches_all_states_and_measured_cycles_for_ten_thousand_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            trained, exported, vectors = (workspace / name for name in ("trained", "exported", "vectors"))
            commands = [
                ["model.train", "--output-dir", str(trained), "--epochs", "1", "--seed", "17",
                 "--train-trajectories", "2", "--validation-trajectories", "1", "--length", "64"],
                ["model.export_integer", "--checkpoint", str(trained / "checkpoint.pt"),
                 "--output-dir", str(exported)],
                ["model.export_vectors", "--export-dir", str(exported),
                 "--output-dir", str(vectors), "--length", "10000"],
            ]
            for command in commands:
                result = subprocess.run([PYTHON, "-m", *command], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            before_export, before_vectors = hashes(exported), hashes(vectors)
            bench = workspace / "cfc_core_long_tb.sv"
            bench.write_text(core_testbench(exported, vectors))
            simulation = workspace / "cfc_core_long.vvp"
            compiled = subprocess.run(
                [executable("iverilog"), "-g2012", "-s", "cfc_core_long_tb",
                 "-o", str(simulation), "rtl/cfc_core.sv", str(bench)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            simulated = subprocess.run([executable("vvp"), str(simulation)], cwd=ROOT,
                                       capture_output=True, text=True, timeout=120)
            self.assertEqual(simulated.returncode, 0, simulated.stdout + simulated.stderr)
            self.assertIn("PASS: 10000 continuous steps", simulated.stdout)
            self.assertEqual(hashes(exported), before_export)
            self.assertEqual(hashes(vectors), before_vectors)


if __name__ == "__main__":
    unittest.main()
