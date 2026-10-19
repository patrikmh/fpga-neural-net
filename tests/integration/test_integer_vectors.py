"""Long continuous integer-vector export tracer, independent of RTL."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from integration.test_model_integer import half_away, hex_words, independent_step, saturate
from model.data import signal_trajectories


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IntegerVectorTracerTests(unittest.TestCase):
    def run_cli(self, *arguments):
        result = subprocess.run([PYTHON, "-m", *arguments], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0,
                         "Integer-vector CLI pipeline must complete:\n" + result.stdout + result.stderr)

    def test_vector_export_matches_ten_thousand_continuous_independent_integer_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            trained, exported, vectors = (workspace / name for name in ("trained", "exported", "vectors"))
            self.run_cli("model.train", "--output-dir", str(trained), "--epochs", "1",
                         "--seed", "17", "--train-trajectories", "2",
                         "--validation-trajectories", "1", "--length", "64")
            self.run_cli("model.export_integer", "--checkpoint", str(trained / "checkpoint.pt"),
                         "--output-dir", str(exported))
            self.run_cli("model.export_vectors", "--export-dir", str(exported),
                         "--output-dir", str(vectors), "--length", "10000")

            artifact = json.loads((exported / "manifest.json").read_text())
            manifest = json.loads((vectors / "manifest.json").read_text())
            self.assertEqual(manifest["count"], 10000)
            self.assertEqual(manifest["seed"], 17)
            self.assertEqual(manifest["initial_hidden"], [0] * 8)
            self.assertEqual(manifest["numeric"], artifact["numeric"])
            self.assertEqual(manifest["export_manifest_sha256"], sha256(exported / "manifest.json"))
            self.assertEqual(manifest["checkpoint_sha256"], sha256(trained / "checkpoint.pt"))
            self.assertEqual(manifest["vectors_sha256"], sha256(vectors / "vectors.hex"))
            rows = (vectors / "vectors.hex").read_text().splitlines()
            self.assertEqual(len(rows), 10000)
            self.assertTrue(all(len(row) == 40 and all(c in "0123456789abcdefABCDEF" for c in row)
                                for row in rows), "Each vector must contain ten 16-bit hexadecimal words")
            decoded = [
                [int(row[offset:offset + 4], 16) -
                 (65536 if int(row[offset:offset + 4], 16) >= 32768 else 0)
                 for offset in range(0, 40, 4)]
                for row in rows
            ]
            trajectory = signal_trajectories(17, "validation", 1, 10001)[0]
            mean, std = (artifact["normalization"][key] for key in ("mean", "std"))
            expected_inputs = [saturate(half_away((float(value) - mean) / std * 8192))
                               for value in trajectory[:-1]]
            self.assertEqual([row[0] for row in decoded], expected_inputs)
            weights, tanh, sigmoid = (hex_words(exported / filename)
                                     for filename in ("weights.hex", "tanh.hex", "sigmoid.hex"))
            for filename in ("weights.hex", "tanh.hex", "sigmoid.hex"):
                self.assertEqual(artifact["files"][filename], sha256(exported / filename))
            hidden = [0] * 8
            for step, row in enumerate(decoded):
                hidden, output = independent_step(row[0], hidden, weights, tanh, sigmoid)
                self.assertEqual(row[1:], hidden + [output], f"Continuous recurrence differs at step {step}")


if __name__ == "__main__":
    unittest.main()
