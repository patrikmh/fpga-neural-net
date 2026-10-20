"""Causality, quaternion semantics, and train-only normalization contracts."""

# Behaviors covered:
# 1. Future sensor inputs cannot alter earlier quaternion predictions.
# 2. Passing hidden state across chunks reproduces a continuous full trial.
# 3. Quaternion loss and angular errors identify q and -q as the same rotation.
# 4. Invalid NaN targets do not affect loss or produce invalid gradients.
# 5. Population normalization includes only train sensors, regardless of validation.

import importlib
import importlib.util
import math
import unittest

import numpy as np
import torch


class BroadContractTests(unittest.TestCase):
    def module(self, name):
        self.assertIsNotNone(importlib.util.find_spec(name), f"Required BROAD module must exist: {name}")
        return importlib.import_module(name)

    def test_future_sensor_changes_preserve_earlier_predictions(self):
        network_class = self.module("model.broad_network").BroadPredictor
        with torch.random.fork_rng():
            torch.manual_seed(11)
            network = network_class().eval()
            inputs = torch.randn(2, 20, 9)
            changed = inputs.clone()
            changed[:, 10:] += 100
            with torch.no_grad():
                before, _ = network(inputs)
                after, _ = network(changed)
        torch.testing.assert_close(before[:, :10], after[:, :10], rtol=0, atol=0)

    def test_hidden_state_passed_between_chunks_matches_continuous_trial(self):
        network_class = self.module("model.broad_network").BroadPredictor
        with torch.random.fork_rng():
            torch.manual_seed(13)
            network = network_class().eval()
            inputs = torch.randn(2, 23, 9)
            with torch.no_grad():
                full, final = network(inputs)
                first, hidden = network(inputs[:, :7])
                second, chunk_final = network(inputs[:, 7:], hidden)
        torch.testing.assert_close(full, torch.cat((first, second), dim=1), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(final, chunk_final, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(torch.linalg.vector_norm(full, dim=-1), torch.ones(2, 23), atol=1e-6, rtol=0)

    def test_quaternion_sign_does_not_change_loss_or_full_orientation_error(self):
        training = self.module("model.broad_train")
        prediction = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]])
        target = torch.tensor([[1., 0, 0, 0], [math.sqrt(0.5), math.sqrt(0.5), 0, 0]])
        mask = torch.tensor([True, True])
        torch.testing.assert_close(training.quaternion_loss(prediction, target, mask),
                                   training.quaternion_loss(prediction, -target, mask))
        torch.testing.assert_close(training.orientation_errors_deg(prediction, target),
                                   torch.tensor([0., 90.]), atol=1e-4, rtol=0)
        torch.testing.assert_close(training.orientation_errors_deg(prediction, target),
                                   training.orientation_errors_deg(prediction, -target))

    def test_invalid_nan_targets_have_no_loss_effect_and_finite_zero_invalid_gradients(self):
        training = self.module("model.broad_train")
        prediction = torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]], requires_grad=True)
        target = torch.tensor([[math.sqrt(0.5), math.sqrt(0.5), 0, 0], [float("nan")] * 4])
        mask = torch.tensor([True, False])
        loss = training.quaternion_loss(prediction, target, mask)
        expected = training.quaternion_loss(prediction[:1], target[:1], mask[:1])
        torch.testing.assert_close(loss, expected)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad[0].abs().sum()), 0)
        torch.testing.assert_close(prediction.grad[1], torch.zeros(4), rtol=0, atol=0)

    def test_normalization_uses_population_moments_of_training_sensors_only(self):
        fit_normalization = self.module("model.broad_train").fit_normalization
        a = np.arange(27, dtype=np.float64).reshape(3, 9)
        b = np.arange(18, dtype=np.float64).reshape(2, 9) + 100
        trials = [{"split": "train", "sensors": a}, {"split": "train", "sensors": b},
                  {"split": "validation", "sensors": np.full((4, 9), 1e9)}]
        normalization = fit_normalization(trials)
        trials[-1]["sensors"] *= -1000
        changed = fit_normalization(trials)
        expected = np.concatenate((a, b))
        np.testing.assert_allclose(normalization["mean"], expected.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(normalization["std"], expected.std(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(normalization["mean"], changed["mean"])
        np.testing.assert_array_equal(normalization["std"], changed["std"])


if __name__ == "__main__":
    unittest.main()
