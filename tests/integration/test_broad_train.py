"""Real CPU BROAD training tracer on bound synthetic train/validation trials."""

import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from ncps.torch import CfCCell


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_fixture(directory):
    trials, inputs, outputs = [], {}, {}
    for trial in range(1, 4):
        split = "train" if trial < 3 else "validation"
        t = np.arange(24, dtype=np.float64)
        sensors = np.column_stack([np.sin(t / (column + 2) + trial) + column + trial * 0.1
                                   for column in range(9)]).astype(np.float32)
        if split == "validation":
            sensors += 1000
        angle = (t * 0.03 + trial * 0.02) / 2
        quaternion = np.column_stack((np.cos(angle), np.sin(angle), np.zeros((24, 2)))).astype(np.float32)
        valid = np.ones(24, dtype=bool)
        valid[7] = False
        quaternion[7] = [1, 0, 0, 0]
        movement = np.ones(24, dtype=bool)
        movement[0] = False
        filename = f"{split}/{trial:02d}.npz"
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sensors=sensors, quaternion=quaternion, movement=movement,
                 reference_valid=valid, sample_index=np.arange(24))
        original = f"hdf5/trial{trial:02d}.hdf5"
        inputs[original] = hashlib.sha256(original.encode()).hexdigest()
        outputs[filename] = sha256(path)
        trials.append({"id": trial, "file": original, "rate_hz": 200.0, "split": split,
                       "sha256": inputs[original]})
    spec = {"source_revision": "7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad", "trials": trials}
    spec_text = json.dumps(spec)
    manifest = {"spec": spec, "spec_sha256": hashlib.sha256(spec_text.encode()).hexdigest(),
                "inputs": inputs, "outputs": outputs}
    (directory / "manifest.json").write_text(json.dumps(manifest))


class BroadTrainingTracerTests(unittest.TestCase):
    def test_real_training_updates_cfc_and_reloads_full_validation_orientation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, output = workspace / "prepared", workspace / "trained"
            prepared_fixture(prepared)
            manifest_hash = sha256(prepared / "manifest.json")
            result = subprocess.run(
                [PYTHON, "-m", "model.broad_train", "--prepared-dir", str(prepared),
                 "--output-dir", str(output), "--epochs", "1", "--updates-per-epoch", "2",
                 "--batch-size", "2", "--window", "16", "--warmup", "2",
                 "--validation-every", "1", "--seed", "7"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0,
                             "Actual BROAD training CLI must complete:\n" + result.stdout + result.stderr)
            from model.broad_network import BroadPredictor

            checkpoint = torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(checkpoint["config"], report["config"])
            self.assertEqual(checkpoint["prepared_manifest_sha256"], manifest_hash)
            self.assertEqual(sha256(prepared / "manifest.json"), manifest_hash)
            self.assertEqual(checkpoint["provenance"]["sources"]["ncps/torch/cfc_cell.py"],
                             sha256(Path(inspect.getfile(CfCCell))))
            torch.manual_seed(checkpoint["config"]["seed"])
            network = BroadPredictor()
            self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 612)
            self.assertEqual(report["parameter_count"], 612)
            initial = {name: value.detach().clone() for name, value in network.named_parameters()}
            network.load_state_dict(checkpoint["state_dict"])
            network.eval()
            for component in ("cell", "readout"):
                self.assertTrue(any(not torch.equal(value, initial[name])
                                    for name, value in network.named_parameters()
                                    if name.startswith(component + ".")), f"Training must update {component}")
            self.assertTrue(math.isfinite(report["training_loss"]))
            training_sensors = []
            for trial in (1, 2):
                with np.load(prepared / f"train/{trial:02d}.npz", allow_pickle=False) as trial_data:
                    training_sensors.append(trial_data["sensors"])
            concatenated = np.concatenate(training_sensors).astype(np.float64)
            normalization = checkpoint["normalization"]
            np.testing.assert_allclose(normalization["mean"], concatenated.mean(axis=0), rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(normalization["std"], concatenated.std(axis=0), rtol=1e-7, atol=1e-8)
            self.assertEqual(len(report["validation"]), 1)
            measured = []
            for entry in report["validation"]:
                self.assertEqual(entry["trial_id"], 3)
                with np.load(output / entry["predictions_file"], allow_pickle=False) as saved:
                    with np.load(prepared / "validation/03.npz", allow_pickle=False) as source:
                        np.testing.assert_array_equal(saved["sensors"], source["sensors"])
                        np.testing.assert_array_equal(saved["targets"], source["quaternion"])
                        for name in ("movement", "reference_valid", "sample_index"):
                            np.testing.assert_array_equal(saved[name], source[name])
                    predictions = saved["predictions"]
                    self.assertEqual(predictions.shape, (24, 4))
                    np.testing.assert_allclose(np.linalg.norm(predictions, axis=1), 1, atol=1e-6)
                    normalized = (saved["sensors"] - np.asarray(normalization["mean"])) / np.asarray(normalization["std"])
                    with torch.no_grad():
                        reloaded, _ = network(torch.tensor(normalized[None], dtype=torch.float32))
                    np.testing.assert_allclose(predictions, reloaded.numpy()[0], rtol=1e-5, atol=1e-6)
                    targets = saved["targets"].astype(np.float64)
                    predicted = predictions.astype(np.float64)
                    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
                    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
                    errors = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(predicted * targets, axis=1)), 0, 1)))
                    mask = saved["movement"] & saved["reference_valid"]
                    mask[:2] = False
                    expected = float(np.sqrt(np.mean(errors[mask] ** 2)))
                    self.assertEqual(entry["samples_scored"], int(mask.sum()))
                    self.assertAlmostEqual(entry["rmse_deg"], expected, delta=1e-3)
                    measured.append(expected)
            self.assertAlmostEqual(report["validation_rmse_deg"], float(np.mean(measured)), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
