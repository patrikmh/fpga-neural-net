"""Behavioral contracts for deterministic data and recurrent training."""

# Behaviors covered:
# 1. Data splits are independent and trajectory prefixes survive count changes.
# 2. Future input cannot affect earlier predictions.
# 3. Hidden state is isolated between trajectories and forward calls.
# 4. One-epoch training is reproducible and independent of validation count.

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from model.data import signal_trajectories
from model.network import SignalPredictor


ROOT = Path(__file__).resolve().parents[1]


class ModelContractTests(unittest.TestCase):
    def test_signal_splits_are_independent_and_preserve_trajectory_count_prefixes(self):
        training = signal_trajectories(7, "train", 2, 64)
        validation = signal_trajectories(7, "validation", 3, 64)
        expanded_training = signal_trajectories(7, "train", 4, 64)
        repeated_validation = signal_trajectories(7, "validation", 1, 64)

        np.testing.assert_array_equal(training, expanded_training[:2])
        np.testing.assert_array_equal(validation[:1], repeated_validation)
        self.assertFalse(np.array_equal(training, validation[:2]))

    def test_changed_future_inputs_do_not_change_earlier_predictions(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            network = SignalPredictor().eval()
            original = torch.randn(2, 24, 1)
            changed = original.clone()
            changed[:, 12:] += 20
            with torch.no_grad():
                before = network(original)
                after = network(changed)

        torch.testing.assert_close(before[:, :12], after[:, :12], rtol=0, atol=0)

    def test_predictions_are_isolated_between_trajectories_and_forward_calls(self):
        with torch.random.fork_rng():
            torch.manual_seed(23)
            network = SignalPredictor().eval()
            inputs = torch.randn(2, 24, 1)
            with torch.no_grad():
                batched = network(inputs)
                first = network(inputs[:1])
                second = network(inputs[1:])
                first_again = network(inputs[:1])

        torch.testing.assert_close(batched, torch.cat((first, second)), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(first, first_again, rtol=0, atol=0)

    def test_one_epoch_training_is_reproducible_and_uses_only_training_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoints, reports, predictions = [], [], []
            for index, validation_count in enumerate((1, 1, 2)):
                output = Path(directory) / str(index)
                result = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), "-m", "model.train",
                     "--output-dir", str(output), "--epochs", "1", "--seed", "7",
                     "--train-trajectories", "2", "--validation-trajectories",
                     str(validation_count), "--length", "64"],
                    cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                checkpoints.append(torch.load(output / "checkpoint.pt", weights_only=True, map_location="cpu"))
                reports.append(json.loads((output / "report.json").read_text()))
                with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
                    predictions.append(saved["predictions"].copy())

            training = signal_trajectories(7, "train", 2, 64)
            expected_normalization = {
                "mean": float(np.mean(training, dtype=np.float64)),
                "std": float(np.std(training, dtype=np.float64)),
            }
            for index, checkpoint in enumerate(checkpoints):
                with self.subTest(run=index):
                    self.assertEqual(checkpoint["normalization"], expected_normalization)
                    self.assertEqual(set(checkpoint["state_dict"]), set(checkpoints[0]["state_dict"]))
                    for name, value in checkpoint["state_dict"].items():
                        self.assertTrue(torch.equal(value, checkpoints[0]["state_dict"][name]), name)
            np.testing.assert_array_equal(predictions[0], predictions[1])
            for metric in ("training_loss", "validation_mse", "persistence_mse", "linear_mse"):
                with self.subTest(metric=metric):
                    self.assertEqual(reports[0][metric], reports[1][metric])


if __name__ == "__main__":
    unittest.main()
