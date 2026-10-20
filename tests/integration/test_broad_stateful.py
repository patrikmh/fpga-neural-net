"""Stateful training CLI tracer against an independent chronological TBPTT run."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from ncps.torch import CfCCell

from integration.test_broad_train import prepared_fixture
from model.broad_network import BroadPredictor


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orientation_score(network, validation):
    scores = []
    network.eval()
    with torch.no_grad():
        for trial in validation:
            prediction, _ = network(torch.tensor(trial["sensors"][None], dtype=torch.float32))
            p = prediction.numpy()[0].astype(np.float64)
            q = trial["quaternion"].astype(np.float64)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            errors = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(p * q, axis=1)), 0, 1)))
            mask = trial["movement"] & trial["reference_valid"]
            mask[:2] = False
            scores.append(float(np.sqrt(np.mean(errors[mask] ** 2))))
    return float(np.mean(scores))


def reference_training(initial, training, reset_each_chunk=False):
    network = BroadPredictor()
    network.load_state_dict(initial)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=.003)
    steps = 0
    for trial_index in np.random.default_rng(7).permutation(len(training)):
        trial = training[trial_index]
        hidden = None
        for start in range(0, len(trial["sensors"]), 8):
            if reset_each_chunk:
                hidden = None
            predicted, hidden = network(torch.tensor(trial["sensors"][None, start:start + 8], dtype=torch.float32), hidden)
            mask = (trial["movement"][start:start + 8] & trial["reference_valid"][start:start + 8]).copy()
            mask[np.arange(start, min(start + 8, len(trial["sensors"]))) < 2] = False
            if mask.any():
                selected = predicted[0, torch.tensor(mask)]
                targets = torch.tensor(trial["quaternion"][start:start + 8][mask], dtype=torch.float32)
                selected = selected / torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
                targets = targets / torch.linalg.vector_norm(targets, dim=-1, keepdim=True)
                dot = (selected * targets).sum(dim=-1).clamp(-1, 1)
                loss = (1 - dot.square()).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                steps += 1
            hidden = hidden.detach()
    return network, steps


class BroadStatefulTrainingTracerTests(unittest.TestCase):
    def test_stateful_cli_matches_chronological_detached_hidden_training_and_best_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared = workspace / "prepared"
            prepared_fixture(prepared)
            manifest_path = prepared / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            loaded = []
            first_sensors = None
            for filename in sorted(manifest["outputs"]):
                path = prepared / filename
                with np.load(path, allow_pickle=False) as source:
                    trial = {name: source[name].copy() for name in source.files}
                if first_sensors is None:
                    first_sensors = trial["sensors"].copy()
                if filename.startswith("validation/"):
                    trial["sensors"] = first_sensors + .02
                trial["quaternion"][:] = [1, 0, 0, 0]
                trial["movement"][:] = True
                trial["reference_valid"][:] = True
                trial["reference_valid"][8:16] = False
                np.savez(path, **trial)
                manifest["outputs"][filename] = sha256(path)
                trial["split"] = filename.split("/")[0]
                loaded.append(trial)
            manifest_path.write_text(json.dumps(manifest))
            training = [trial for trial in loaded if trial["split"] == "train"]
            validation = [trial for trial in loaded if trial["split"] == "validation"]
            torch.set_num_threads(1)
            torch.manual_seed(7)
            network = BroadPredictor()
            with torch.no_grad():
                network.readout.weight.zero_()
                network.readout.bias.copy_(torch.tensor([2 ** -.5, 0, 0, 2 ** -.5]))
            initial = {name: value.detach().clone() for name, value in network.state_dict().items()}
            sources = {f"model/{name}": sha256(ROOT / "model" / name)
                       for name in ("broad_network.py", "broad_train.py")}
            sources["ncps/torch/cfc_cell.py"] = sha256(Path(inspect.getfile(CfCCell)))
            checkpoint = {
                "state_dict": initial, "normalization": {"mean": [0.] * 9, "std": [1.] * 9, "constant_channels": []},
                "config": {"seed": 7, "warmup": 2, "input_size": 9, "hidden_size": 8, "output_size": 4,
                           "mode": "default", "backbone_layers": 0, "dt": 1.0,
                           "loss": "1-squared_normalized_quaternion_dot"},
                "prepared_manifest_sha256": sha256(manifest_path),
                "provenance": {"sources": sources,
                               "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                            "numpy": np.__version__, "ncps": importlib.metadata.version("ncps")}},
                "best_epoch": 0,
            }
            initial_path = workspace / "initial.pt"
            torch.save(checkpoint, initial_path)
            initial_score = orientation_score(network, validation)
            reference, expected_steps = reference_training(initial, training)
            reference_score = orientation_score(reference, validation)
            broken, _ = reference_training(initial, training, reset_each_chunk=True)
            self.assertTrue(any(not torch.equal(value, broken.state_dict()[name])
                                for name, value in reference.state_dict().items()),
                            "Fixture must detect resets at chunk boundaries")
            self.assertLess(reference_score, initial_score, "Fixture must select an observable trained checkpoint")
            self.assertEqual(expected_steps, 4)
            output = workspace / "stateful"
            result = subprocess.run(
                [PYTHON, "-m", "model.broad_train_stateful", "--prepared-dir", str(prepared),
                 "--initial-checkpoint", str(initial_path), "--initial-checkpoint-sha256", sha256(initial_path),
                 "--output-dir", str(output), "--epochs", "1", "--chunk-size", "8",
                 "--learning-rate", "0.003", "--seed", "7"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0,
                             "Stateful chronological training CLI must complete:\n" + result.stdout + result.stderr)
            saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["normalization"], checkpoint["normalization"])
            self.assertEqual(saved["provenance"], checkpoint["provenance"])
            self.assertEqual(saved["prepared_manifest_sha256"], sha256(manifest_path))
            self.assertEqual(saved["initial_checkpoint_sha256"], sha256(initial_path))
            recipe = {"model/broad_train_stateful.py": sha256(ROOT / "model/broad_train_stateful.py")}
            self.assertEqual(saved["training_recipe_sources"], recipe)
            self.assertEqual(report["training_recipe_sources"], recipe)
            self.assertEqual(report["optimizer_steps"], expected_steps)
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(saved["best_epoch"], 1)
            self.assertAlmostEqual(report["initial_validation_rmse_deg"], initial_score, delta=1e-4)
            self.assertAlmostEqual(report["validation_rmse_deg"], reference_score, delta=1e-4)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(saved["state_dict"][name], expected, rtol=1e-6, atol=1e-7, msg=name)


def initial_fixture(workspace, warmup=2, identity=False):
    prepared = workspace / "prepared"
    prepared_fixture(prepared)
    manifest_path = prepared / "manifest.json"
    if identity:
        manifest = json.loads(manifest_path.read_text())
        for filename in manifest["outputs"]:
            path = prepared / filename
            with np.load(path, allow_pickle=False) as source:
                trial = {name: source[name].copy() for name in source.files}
            trial["quaternion"][:] = [1, 0, 0, 0]
            np.savez(path, **trial)
            manifest["outputs"][filename] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
    torch.manual_seed(7)
    network = BroadPredictor()
    if identity:
        with torch.no_grad():
            network.readout.weight.zero_()
            network.readout.bias.copy_(torch.tensor([1., 0, 0, 0]))
    sources = {f"model/{name}": sha256(ROOT / "model" / name)
               for name in ("broad_network.py", "broad_train.py")}
    sources["ncps/torch/cfc_cell.py"] = sha256(Path(inspect.getfile(CfCCell)))
    checkpoint = {
        "state_dict": network.state_dict(),
        "normalization": {"mean": [0.] * 9, "std": [1.] * 9, "constant_channels": []},
        "config": {"seed": 7, "warmup": warmup, "input_size": 9, "hidden_size": 8,
                   "output_size": 4, "mode": "default", "backbone_layers": 0, "dt": 1.},
        "prepared_manifest_sha256": sha256(manifest_path),
        "provenance": {"sources": sources,
                       "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                    "numpy": np.__version__, "ncps": importlib.metadata.version("ncps")}},
        "best_epoch": 0,
    }
    checkpoint_path = workspace / "initial.pt"
    torch.save(checkpoint, checkpoint_path)
    return prepared, checkpoint_path, checkpoint


# S2 behaviors: reject stale inputs before reading trials; preserve nonempty
# output; apply warmup across chunk boundaries; retain an equally good epoch 0.
class BroadStatefulGuardTests(unittest.TestCase):
    def run_training(self, prepared, initial, output, initial_sha=None):
        return subprocess.run(
            [PYTHON, "-m", "model.broad_train_stateful", "--prepared-dir", str(prepared),
             "--initial-checkpoint", str(initial), "--initial-checkpoint-sha256", initial_sha or sha256(initial),
             "--output-dir", str(output), "--epochs", "1", "--chunk-size", "8",
             "--learning-rate", ".003", "--seed", "7"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )

    def test_initial_identity_and_configuration_guards_precede_missing_trial_access(self):
        for cause in ("sha", "provenance", "config", "manifest"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                prepared, initial, checkpoint = initial_fixture(workspace)
                if cause == "provenance":
                    checkpoint["provenance"]["sources"]["model/broad_network.py"] = "0" * 64
                elif cause == "config":
                    checkpoint["config"]["dt"] = .5
                elif cause == "manifest":
                    checkpoint["prepared_manifest_sha256"] = "0" * 64
                torch.save(checkpoint, initial)
                for path in prepared.rglob("*.npz"):
                    path.unlink()
                output = workspace / "trained"
                result = self.run_training(prepared, initial, output, "0" * 64 if cause == "sha" else None)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr.lower(), r"(sha256|identity|provenance|configuration)")
                self.assertNotIn("no such file", result.stderr.lower())
                self.assertFalse(output.exists() and any(output.iterdir()))

    def test_nonempty_output_is_rejected_without_changing_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, initial, _ = initial_fixture(workspace)
            output = workspace / "trained"
            output.mkdir()
            sentinel = {"report.json": b"old report\n", "checkpoint.pt": b"old checkpoint\n"}
            for filename, contents in sentinel.items():
                (output / filename).write_bytes(contents)
            result = self.run_training(prepared, initial, output)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("empty", result.stderr.lower())
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, sentinel)

    def test_warmup_larger_than_a_chunk_scores_only_after_the_trial_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, initial, _ = initial_fixture(workspace, warmup=20)
            output = workspace / "trained"
            result = self.run_training(prepared, initial, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["config"]["warmup"], 20)
            self.assertEqual(report["optimizer_steps"], 2)
            self.assertEqual(report["validation"][0]["samples_scored"], 4)

    def test_zero_loss_identity_targets_retain_unchanged_epoch_zero_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, initial, checkpoint = initial_fixture(workspace, identity=True)
            output = workspace / "trained"
            result = self.run_training(prepared, initial, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["best_epoch"], 0)
            self.assertEqual(saved["best_epoch"], 0)
            self.assertGreater(report["optimizer_steps"], 0)
            self.assertEqual(report["initial_validation_rmse_deg"], 0)
            self.assertEqual(report["validation_rmse_deg"], 0)
            self.assertEqual(saved["normalization"], checkpoint["normalization"])
            self.assertEqual(saved["provenance"], checkpoint["provenance"])
            for name, value in checkpoint["state_dict"].items():
                self.assertTrue(torch.equal(saved["state_dict"][name], value), name)


if __name__ == "__main__":
    unittest.main()
