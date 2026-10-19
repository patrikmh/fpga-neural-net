"""Real CPU training tracer with independent checkpoint and metric validation."""

import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ModelTrainingTracerTests(unittest.TestCase):
    def test_training_cli_saves_reloadable_cfc_and_reproducible_validation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            result = subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-m", "model.train",
                 "--output-dir", str(output), "--epochs", "2", "--seed", "7",
                 "--train-trajectories", "2", "--validation-trajectories", "1",
                 "--length", "64"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(
                result.returncode, 0,
                "Real CPU training CLI must complete:\n" + result.stdout + result.stderr,
            )

            import numpy as np
            import torch
            from model.network import SignalPredictor

            report = json.loads((output / "report.json").read_text())
            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"], report["config"])
            torch.manual_seed(checkpoint["config"]["seed"])
            network = SignalPredictor()
            initial_parameters = {
                name: value.detach().clone() for name, value in network.named_parameters()
            }
            network.load_state_dict(checkpoint["state_dict"])
            for component in ("cell", "readout"):
                with self.subTest(trained_component=component):
                    self.assertTrue(
                        any(
                            not torch.equal(value, initial_parameters[name])
                            for name, value in network.named_parameters()
                            if name.startswith(component + ".")
                        ),
                        f"Training must update parameters in {component}",
                    )
            network.eval()
            self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 329)
            self.assertEqual(report["parameter_count"], 329)
            self.assertTrue(math.isfinite(report["training_loss"]))
            mean = float(checkpoint["normalization"]["mean"])
            std = float(checkpoint["normalization"]["std"])
            self.assertTrue(math.isfinite(mean))
            self.assertTrue(math.isfinite(std) and std > 0)

            with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
                trajectories = saved["trajectories"]
                current, previous = saved["inputs"], saved["previous"]
                targets, predictions = saved["targets"], saved["predictions"]
                self.assertEqual(trajectories.shape, (1, 64))
                self.assertEqual(predictions.shape, (1, 55))
                np.testing.assert_array_equal(current, trajectories[:, 8:-1])
                np.testing.assert_array_equal(previous, trajectories[:, 7:-2])
                np.testing.assert_array_equal(targets, trajectories[:, 9:])
                self.assertTrue(np.isfinite(predictions).all())
                normalized = torch.tensor(
                    (trajectories[:, :-1, None] - mean) / std, dtype=torch.float32,
                )
                with torch.no_grad():
                    reproduced = network(normalized).cpu().numpy()[:, 8:, 0] * std + mean
                np.testing.assert_allclose(predictions, reproduced, rtol=1e-5, atol=1e-6)

                candidates = {
                    "validation_mse": predictions,
                    "persistence_mse": current,
                    "linear_mse": 2 * current - previous,
                }
                for metric, candidate in candidates.items():
                    with self.subTest(metric=metric):
                        expected = float(np.mean(
                            (candidate.astype(np.float64) - targets.astype(np.float64)) ** 2
                        ))
                        self.assertTrue(math.isfinite(report[metric]))
                        self.assertTrue(math.isclose(report[metric], expected, rel_tol=1e-5, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
