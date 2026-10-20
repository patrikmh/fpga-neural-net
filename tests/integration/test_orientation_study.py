"""One actual CLI tracer for synthetic orientation training and integer validation.

TRACER behavior: generated sensor trajectories become a genuinely trained H96
checkpoint, a strict integer export, and independently replayable validation.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from integration.test_model_integer import half_away, saturate


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrientationStudyTracerTests(unittest.TestCase):
    def run_cli(self, *arguments):
        command = [PYTHON, "-m", *arguments]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0,
                         "Actual orientation pipeline command must complete:\n"
                         + " ".join(command) + "\n" + result.stdout + result.stderr)

    def test_generated_trajectories_train_h96_and_integer_validation_replays_every_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset, trained, exported = (workspace / name for name in ("data", "trained", "export"))
            report_path = workspace / "evaluation/report.json"
            self.run_cli("model.orientation_generate", "--output-dir", str(dataset), "--seed", "17",
                         "--train-sequences", "2", "--validation-sequences", "1", "--length", "40")
            self.run_cli("model.orientation_train", "--dataset-dir", str(dataset), "--output-dir", str(trained),
                         "--epochs", "1", "--hidden-size", "96", "--warmup", "0", "--chunk-size", "16",
                         "--seed", "23", "--learning-rate", "0.001")
            self.run_cli("model.broad_export_integer", "--checkpoint", str(trained / "checkpoint.pt"),
                         "--output-dir", str(exported))
            self.run_cli("model.orientation_evaluate", "--dataset-dir", str(dataset), "--split", "validation",
                         "--export-dir", str(exported), "--output", str(report_path))

            from model.broad_integer import GoldenBroadPredictor
            from model.broad_network import BroadPredictor
            from model.broad_train_stateful import current_provenance

            manifest = json.loads((dataset / "manifest.json").read_text())
            self.assertEqual(manifest["dt_s"], .0035)
            self.assertEqual(len(manifest["sequences"]), 3)
            sequences = []
            for entry in manifest["sequences"]:
                path = dataset / entry["path"]
                self.assertEqual(entry["sha256"], sha256(path))
                with np.load(path, allow_pickle=False) as saved:
                    sensors, quaternion = saved["sensors"].copy(), saved["quaternion"].copy()
                self.assertEqual(sensors.shape, (40, 9))
                self.assertEqual(quaternion.shape, (40, 4))
                self.assertTrue(np.isfinite(sensors).all() and np.isfinite(quaternion).all())
                np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-7)
                np.testing.assert_allclose(sensors[0, 3:6], 0, atol=1e-10)
                self.assertLess(abs(float(quaternion[0, 0])), .999,
                                "Initial pose must not always be identity")
                held = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=1)) > 1 - 1e-12
                self.assertTrue(held.any(), "Short trajectories must include stationary supervision")
                np.testing.assert_allclose(sensors[1:, 3:6][held], 0, atol=1e-7)
                sequences.append((entry, sensors, quaternion))
            training = [item for item in sequences if item[0]["split"] == "train"]
            validation = [item for item in sequences if item[0]["split"] == "validation"]
            self.assertEqual((len(training), len(validation)), (2, 1))
            self.assertEqual(len({str(item[0]["sequence_id"]) for item in sequences}), 3)
            self.assertTrue(all(not np.array_equal(validation[0][1], item[1]) for item in training),
                            "Validation must be its own generated trajectory")

            checkpoint = torch.load(trained / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"]["hidden_size"], 96)
            self.assertEqual(checkpoint["config"]["warmup"], 0)
            self.assertEqual(checkpoint["provenance"], current_provenance())
            for name in ("model/orientation_generate.py", "model/orientation_train.py",
                         "model/orientation_evaluate.py", "host/orientation_sim.py"):
                self.assertEqual(checkpoint["training_recipe_sources"][name], sha256(ROOT / name))
            train_sensors = np.concatenate([item[1] for item in training]).astype(np.float64)
            mean = train_sensors.mean(axis=0)
            std = np.maximum(train_sensors.std(axis=0), 1e-8)
            np.testing.assert_allclose(checkpoint["normalization"]["mean"], mean, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(checkpoint["normalization"]["std"], std, atol=1e-10, rtol=1e-10)
            train_report = json.loads((trained / "report.json").read_text())
            self.assertGreater(train_report["optimizer_steps"], 0)
            self.assertEqual(train_report["training_samples_scored"], 80,
                             "Include stationary frames and the final partial chunk")
            torch.manual_seed(23)
            initial = BroadPredictor(hidden_size=96).state_dict()
            self.assertTrue(any(not torch.equal(initial[name], value)
                                for name, value in checkpoint["state_dict"].items()),
                            "Checkpoint must contain trained weights, not an untouched initialization")

            predictor = GoldenBroadPredictor.from_export(exported)
            self.assertEqual(predictor.hidden_size, 96)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["split"], "validation")
            self.assertEqual(report["sample_count"], 40)
            self.assertEqual(len(report["sequences"]), len(validation))
            all_errors = []
            for result, (entry, sensors, targets) in zip(report["sequences"], validation):
                self.assertEqual(result["sequence_id"], entry["sequence_id"])
                self.assertEqual(result["sample_count"], 40)
                quantized = np.array([[saturate(half_away(value * 2048)) for value in row]
                                      for row in (sensors.astype(np.float64) - mean) / std], dtype=np.int16)
                with np.load(report_path.parent / result["predictions_file"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["targets"], targets)
                    np.testing.assert_array_equal(saved["quantized_inputs"], quantized)
                    self.assertEqual(saved["hidden"].shape, (40, 96))
                    self.assertEqual(saved["raw_outputs"].shape, (40, 4))
                    for name in ("quantized_inputs", "hidden", "raw_outputs"):
                        self.assertEqual(saved[name].dtype, np.dtype("int16"))
                    predictor.reset()
                    for index, inputs in enumerate(quantized):
                        hidden, raw = predictor.step(inputs)
                        np.testing.assert_array_equal(saved["hidden"][index], hidden)
                        np.testing.assert_array_equal(saved["raw_outputs"][index], raw)
                    raw = saved["raw_outputs"].astype(np.float64)
                    self.assertTrue(np.all(np.linalg.norm(raw, axis=1) > 0))
                    predicted = raw / np.linalg.norm(raw, axis=1, keepdims=True)
                    np.testing.assert_allclose(saved["predictions"], predicted, atol=1e-7, rtol=1e-6)
                    unit_targets = targets.astype(np.float64)
                    unit_targets /= np.linalg.norm(unit_targets, axis=1, keepdims=True)
                    errors = np.degrees(2 * np.arccos(np.clip(
                        np.abs(np.sum(predicted * unit_targets, axis=1)), 0, 1)))
                    np.testing.assert_allclose(saved["angle_errors_deg"], errors, atol=1e-5, rtol=1e-6)
                    self.assertAlmostEqual(result["full_rmse_deg"], float(np.sqrt(np.mean(errors ** 2))),
                                           delta=1e-5)
                    all_errors.extend(errors.tolist())
            self.assertTrue(np.isfinite(report["full_rmse_deg"]))
            self.assertAlmostEqual(report["full_rmse_deg"], float(np.sqrt(np.mean(np.square(all_errors)))),
                                   delta=1e-5)


if __name__ == "__main__":
    unittest.main()
