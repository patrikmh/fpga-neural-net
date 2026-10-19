"""Held-out evaluation contracts using tiny disposable seed-17 fixtures."""

# Behaviors covered:
# 1. Metrics align predictions/baselines and use an inclusive quality threshold.
# 2. Evaluation CLI exports independently verifiable, identity-bound artifacts.
# 3. Checkpoint/source identity mismatches are rejected before output artifacts.

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from ncps.torch import CfCCell


ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_calculated_metrics(trajectories, full_predictions, std, warmup):
    targets = trajectories[:, warmup + 1:]
    current = trajectories[:, warmup:-1]
    previous = trajectories[:, warmup - 1:-2]
    candidates = {"cfc": full_predictions[:, warmup:], "persistence": current,
                  "linear": 2 * current - previous}
    raw = {name: float(np.mean((values.astype(np.float64) - targets) ** 2))
           for name, values in candidates.items()}
    return raw, {name: value / std ** 2 for name, value in raw.items()}


class ModelEvaluationTests(unittest.TestCase):
    def fixture(self, workspace):
        training = workspace / "trained"
        result = subprocess.run(
            [PYTHON, "-m", "model.train", "--output-dir", str(training),
             "--seed", "17", "--epochs", "1", "--train-trajectories", "2",
             "--validation-trajectories", "1", "--length", "64"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checkpoint_path = training / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        spec = {
            "checkpoint_sha256": sha256(checkpoint_path), "seed": 17,
            "trajectories": 2, "length": 64, "warmup": 8,
            "generator": checkpoint["config"]["generator"],
            "float_required_improvement": 0.2,
            "integer_epsilon_normalized_mse": 1e-6,
            "sources": {
                "model/data.py": sha256(ROOT / "model/data.py"),
                "model/network.py": sha256(ROOT / "model/network.py"),
                "ncps/torch/cfc_cell.py": sha256(Path(inspect.getfile(CfCCell))),
            },
        }
        return checkpoint_path, checkpoint, spec

    def evaluate(self, checkpoint, spec_path, output):
        return subprocess.run(
            [PYTHON, "-m", "model.evaluate", "--checkpoint", str(checkpoint),
             "--spec", str(spec_path), "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )

    def test_metrics_align_baselines_and_accept_the_exact_improvement_threshold(self):
        self.assertIsNotNone(importlib.util.find_spec("model.evaluate"),
                             "Evaluation metrics module must exist")
        from model.evaluate import metrics

        trajectories = np.array([[0, 1, 2, 5], [10, 11, 14, 15]], dtype=np.float64)
        predictions = np.array([[-999, 4, 7], [-999, 16, 17]], dtype=np.float64)
        report = metrics(trajectories, predictions, std=2, warmup=1, required_improvement=0.2)
        raw, normalized = independently_calculated_metrics(trajectories, predictions, 2, 1)
        self.assertEqual(report["raw_mse"], raw)
        self.assertEqual(report["normalized_mse"], normalized)
        self.assertTrue(report["meets_float_target"])
        self.assertEqual(len(report["per_trajectory"]), 2)
        for index, trajectory in enumerate(report["per_trajectory"]):
            expected_raw, expected_normalized = independently_calculated_metrics(
                trajectories[index:index + 1], predictions[index:index + 1], 2, 1,
            )
            self.assertEqual(trajectory["raw_mse"], expected_raw)
            self.assertEqual(trajectory["normalized_mse"], expected_normalized)
        outside = predictions.copy()
        outside[:, 1:] += 1e-6
        self.assertFalse(metrics(trajectories, outside, 2, 1, 0.2)["meets_float_target"])

    def test_evaluation_cli_exports_reproducible_metrics_and_bound_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            checkpoint_path, checkpoint, spec = self.fixture(workspace)
            spec_path = workspace / "spec.json"
            spec_path.write_text(json.dumps(spec))
            output = workspace / "evaluated"
            result = self.evaluate(checkpoint_path, spec_path, output)
            self.assertEqual(result.returncode, 0,
                             "Evaluation CLI must succeed:\n" + result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["spec"], spec)
            self.assertEqual(report["spec_sha256"], sha256(spec_path))
            self.assertEqual(report["checkpoint_sha256"], sha256(checkpoint_path))
            self.assertEqual(report["sources"], spec["sources"])
            self.assertTrue(report["versions"])
            from model.data import signal_trajectories
            from model.network import SignalPredictor

            network = SignalPredictor().eval()
            network.load_state_dict(checkpoint["state_dict"])
            mean, std = (checkpoint["normalization"][key] for key in ("mean", "std"))
            with np.load(output / "test_predictions.npz", allow_pickle=False) as saved:
                trajectories = saved["trajectories"]
                self.assertEqual(trajectories.shape, (2, 64))
                np.testing.assert_array_equal(trajectories, signal_trajectories(17, "test", 2, 64))
                np.testing.assert_array_equal(saved["inputs"], trajectories[:, 8:-1])
                np.testing.assert_array_equal(saved["previous"], trajectories[:, 7:-2])
                np.testing.assert_array_equal(saved["targets"], trajectories[:, 9:])
                with torch.no_grad():
                    full = network(torch.tensor((trajectories[:, :-1, None] - mean) / std,
                                                dtype=torch.float32)).numpy()[:, :, 0] * std + mean
                np.testing.assert_allclose(saved["predictions"], full[:, 8:], rtol=1e-5, atol=1e-6)
                raw, normalized = independently_calculated_metrics(trajectories, full, std, 8)
                for name in raw:
                    self.assertAlmostEqual(report["raw_mse"][name], raw[name], delta=1e-7)
                    self.assertAlmostEqual(report["normalized_mse"][name], normalized[name], delta=1e-7)
                self.assertEqual(report["meets_float_target"], raw["cfc"] <= 0.8 * raw["persistence"])

    def test_identity_mismatch_is_rejected_before_evaluation_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            checkpoint_path, _, spec = self.fixture(workspace)
            for mismatch in ("checkpoint", "source"):
                with self.subTest(mismatch=mismatch):
                    bad = copy.deepcopy(spec)
                    if mismatch == "checkpoint":
                        bad["checkpoint_sha256"] = "0" * 64
                    else:
                        bad["sources"]["model/data.py"] = "0" * 64
                    spec_path = workspace / f"{mismatch}.json"
                    spec_path.write_text(json.dumps(bad))
                    output = workspace / mismatch
                    result = self.evaluate(checkpoint_path, spec_path, output)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stderr.lower(), r"(hash|sha256|identity|mismatch)")
                    self.assertFalse(output.exists() and any(output.iterdir()),
                                     "Identity rejection must precede output artifacts")


if __name__ == "__main__":
    unittest.main()
