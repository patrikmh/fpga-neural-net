"""Real integer export/evaluation tracer with independent arithmetic checks."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def half_away(value):
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def saturate(value):
    return min(32767, max(-32768, value))


def shift(value, bits):
    magnitude = (abs(value) + (1 << (bits - 1))) >> bits
    return magnitude if value >= 0 else -magnitude


def hex_words(path):
    words = path.read_text().split()
    if not all(len(word) == 4 and all(c in "0123456789abcdefABCDEF" for c in word) for word in words):
        raise AssertionError(f"Expected four-digit hexadecimal words in {path.name}")
    return [int(word, 16) - (65536 if int(word, 16) >= 32768 else 0) for word in words]


def independent_step(x, old, weights, tanh, sigmoid):
    vector = [int(x)] + old
    heads = []
    for head in range(4):
        values = []
        for row in range(8):
            offset = head * 80 + row * 10
            values.append((weights[offset + 9] << 13)
                          + sum(vector[column] * weights[offset + column] for column in range(9)))
        heads.append(values)
    lookup = lambda table, acc: table[min(4095, max(0, shift(acc, 18) + 2048))]
    new = []
    for neuron in range(8):
        f1 = lookup(tanh, heads[0][neuron])
        f2 = lookup(tanh, heads[1][neuron])
        gate = lookup(sigmoid, heads[2][neuron] + heads[3][neuron])
        new.append(saturate(f1 + shift((f2 - f1) * gate, 13)))
    output_acc = (weights[328] << 13) + sum(new[index] * weights[320 + index] for index in range(8))
    return new, saturate(shift(output_acc, 13))


class IntegerModelTracerTests(unittest.TestCase):
    def run_cli(self, *arguments):
        result = subprocess.run([PYTHON, "-m", *arguments], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0,
                         "Model CLI must complete:\n" + result.stdout + result.stderr)

    def test_integer_export_replays_validation_and_reports_independent_precision_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            trained, exported, evaluated = (workspace / name for name in ("trained", "integer", "evaluated"))
            self.run_cli("model.train", "--output-dir", str(trained), "--epochs", "1",
                         "--seed", "17", "--train-trajectories", "2",
                         "--validation-trajectories", "1", "--length", "64")
            checkpoint_path = trained / "checkpoint.pt"
            self.run_cli("model.export_integer", "--checkpoint", str(checkpoint_path),
                         "--output-dir", str(exported))
            self.run_cli("model.evaluate_integer", "--export-dir", str(exported),
                         "--output-dir", str(evaluated), "--trajectories", "2", "--length", "64")

            from model.integer import GoldenIntegerPredictor
            from model.data import signal_trajectories
            from model.network import SignalPredictor

            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            manifest = json.loads((exported / "manifest.json").read_text())
            self.assertEqual(manifest["numeric"], {
                "word_bits": 16, "fractional_bits": 13, "accumulator_bits": 48,
                "accumulator_fractional_bits": 26, "rounding": "half_away_from_zero",
                "activation_entries": 4096, "activation_fractional_bits": 8,
                "activation_zero_index": 2048,
            })
            self.assertEqual(manifest["checkpoint_sha256"], digest(checkpoint_path))
            self.assertEqual(manifest["normalization"], checkpoint["normalization"])
            self.assertEqual(manifest["config"], checkpoint["config"])
            for filename in ("weights.hex", "tanh.hex", "sigmoid.hex", "float_checkpoint.pt"):
                self.assertEqual(manifest["files"][filename], digest(exported / filename))
            self.assertEqual(digest(exported / "float_checkpoint.pt"), digest(checkpoint_path))
            weights = hex_words(exported / "weights.hex")
            tanh, sigmoid = (hex_words(exported / name) for name in ("tanh.hex", "sigmoid.hex"))
            self.assertEqual((len(weights), len(tanh), len(sigmoid)), (329, 4096, 4096))
            ordered = []
            state = checkpoint["state_dict"]
            for name in ("ff1", "ff2", "time_a", "time_b"):
                for row in range(8):
                    ordered.extend(state[f"cell.{name}.weight"][row].tolist())
                    ordered.append(float(state[f"cell.{name}.bias"][row]))
            ordered.extend(state["readout.weight"][0].tolist())
            ordered.append(float(state["readout.bias"][0]))
            for index in (0, 9, 10, 79, 80, 159, 160, 239, 240, 319, 320, 328):
                self.assertEqual(weights[index], saturate(half_away(ordered[index] * 8192)))
            for index in (0, 1, 1024, 2047, 2048, 2049, 3072, 4095):
                x = (index - 2048) / 256
                self.assertEqual(tanh[index], half_away(math.tanh(x) * 8192))
                self.assertEqual(sigmoid[index], half_away(8192 / (1 + math.exp(-x))))

            report = json.loads((evaluated / "report.json").read_text())
            self.assertEqual(report["split"], "validation")
            self.assertEqual(report["warmup"], 8)
            mean, std = (checkpoint["normalization"][key] for key in ("mean", "std"))
            predictor = GoldenIntegerPredictor.from_export(exported)
            with np.load(evaluated / "validation_predictions.npz", allow_pickle=False) as saved:
                trajectories = saved["trajectories"]
                np.testing.assert_array_equal(trajectories, signal_trajectories(17, "validation", 2, 64))
                self.assertEqual(trajectories.shape, (2, 64))
                self.assertEqual(saved["quantized_inputs"].shape, (2, 63))
                self.assertEqual(saved["hidden"].shape, (2, 63, 8))
                for key in ("quantized_inputs", "hidden", "integer_outputs"):
                    self.assertEqual(saved[key].dtype, np.dtype("int16"))
                for key in ("integer_outputs", "integer_predictions", "float_predictions", "targets"):
                    self.assertEqual(saved[key].shape, (2, 63))
                quantized = np.array([
                    [saturate(half_away((float(value) - mean) / std * 8192)) for value in row[:-1]]
                    for row in trajectories
                ], dtype=np.int16)
                np.testing.assert_array_equal(saved["quantized_inputs"], quantized)
                for trajectory in (0, 1, 0):
                    predictor.reset()
                    old = [0] * 8
                    for step, x in enumerate(quantized[trajectory]):
                        old, y = independent_step(x, old, weights, tanh, sigmoid)
                        self.assertEqual(predictor.step(int(x)), (old, y))
                        np.testing.assert_array_equal(saved["hidden"][trajectory, step], old)
                        self.assertEqual(int(saved["integer_outputs"][trajectory, step]), y)
                np.testing.assert_allclose(saved["integer_predictions"],
                                           saved["integer_outputs"].astype(np.float64) / 8192 * std + mean,
                                           rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(saved["targets"], trajectories[:, 1:])
                network = SignalPredictor().eval()
                network.load_state_dict(state)
                with torch.no_grad():
                    floating = network(torch.tensor((trajectories[:, :-1, None] - mean) / std,
                                                    dtype=torch.float32)).numpy()[:, :, 0] * std + mean
                np.testing.assert_allclose(saved["float_predictions"], floating, rtol=1e-5, atol=1e-6)
                metrics = {}
                for kind in ("float", "integer"):
                    error = (saved[f"{kind}_predictions"][:, 8:].astype(np.float64)
                             - saved["targets"][:, 8:].astype(np.float64))
                    metrics[kind] = float(np.mean(error ** 2) / std ** 2)
                    self.assertAlmostEqual(report[f"normalized_{kind}_mse"], metrics[kind], delta=1e-7)
                expected_pass = metrics["integer"] <= metrics["float"] + max(0.1 * metrics["float"], 1e-6)
                self.assertEqual(report["meets_precision_target"], expected_pass)


if __name__ == "__main__":
    unittest.main()
