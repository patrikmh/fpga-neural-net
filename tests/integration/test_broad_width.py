"""Configurable BROAD width through actual training and frozen evaluation."""

# Behaviors covered:
# 1. A 32-neuron trained checkpoint reloads and evaluates with its saved width.
# 2. A 64-neuron predictor has the expected parameter/state/output dimensions.
# 3. Invalid requested widths are rejected before training output is created.

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from integration.test_broad_train import prepared_fixture
from model.broad_network import BroadPredictor


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BroadWidthTests(unittest.TestCase):
    def test_training_and_frozen_evaluation_reload_the_selected_32_neuron_width(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, trained = workspace / "training", workspace / "trained"
            prepared_fixture(prepared)
            training = subprocess.run(
                [PYTHON, "-m", "model.broad_train", "--prepared-dir", str(prepared),
                 "--output-dir", str(trained), "--hidden-size", "32", "--epochs", "1",
                 "--updates-per-epoch", "2", "--batch-size", "2", "--window", "16",
                 "--warmup", "2", "--validation-every", "1", "--seed", "17"],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(training.returncode, 0,
                             "Configurable-width CPU training must succeed:\n" + training.stdout + training.stderr)
            checkpoint_path = trained / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["config"]["hidden_size"], 32)
            self.assertEqual(checkpoint["prepared_manifest_sha256"], sha256(prepared / "manifest.json"))
            network = BroadPredictor(hidden_size=32).eval()
            network.load_state_dict(checkpoint["state_dict"])
            self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 5508)
            train_report = json.loads((trained / "report.json").read_text())
            self.assertEqual(train_report["parameter_count"], 5508)

            frozen_sources = dict(checkpoint["provenance"]["sources"])
            for filename in ("model/broad_evaluate.py", "experiments/broad_prepare.py"):
                frozen_sources[filename] = sha256(ROOT / filename)
            raw_name = "hdf5/synthetic_test07.hdf5"
            raw_hash = hashlib.sha256(raw_name.encode()).hexdigest()
            spec = {
                "source_revision": "7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad",
                "checkpoint_sha256": sha256(checkpoint_path),
                "training_manifest_sha256": checkpoint["prepared_manifest_sha256"],
                "selection_sha256": hashlib.sha256(b"synthetic-width32-selection").hexdigest(),
                "sources": frozen_sources, "versions": checkpoint["provenance"]["versions"],
                "warmup": 2, "baseline_tau_s": .5, "heading_epsilon": 1e-12,
                "trials": [{"id": 7, "split": "test", "file": raw_name,
                            "sha256": raw_hash, "rate_hz": 200.}],
            }
            spec_path = workspace / "frozen.json"
            spec_path.write_text(json.dumps(spec))
            testing = workspace / "testing"
            (testing / "test").mkdir(parents=True)
            shutil.copyfile(prepared / "validation/03.npz", testing / "test/07.npz")
            manifest = {
                "spec": spec, "spec_sha256": sha256(spec_path), "inputs": {raw_name: raw_hash},
                "outputs": {"test/07.npz": sha256(testing / "test/07.npz")},
                "sources": {"experiments/broad_prepare.py": frozen_sources["experiments/broad_prepare.py"]},
            }
            (testing / "manifest.json").write_text(json.dumps(manifest))
            evaluated = workspace / "evaluated"
            evaluation = subprocess.run(
                [PYTHON, "-m", "model.broad_evaluate", "--checkpoint", str(checkpoint_path),
                 "--prepared-dir", str(testing), "--spec", str(spec_path), "--output-dir", str(evaluated)],
                cwd=ROOT, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(evaluation.returncode, 0,
                             "Evaluation must restore checkpoint width:\n" + evaluation.stdout + evaluation.stderr)
            report = json.loads((evaluated / "report.json").read_text())
            self.assertEqual(report["checkpoint_sha256"], spec["checkpoint_sha256"])
            self.assertEqual(len(report["trials"]), 1)
            with np.load(evaluated / report["trials"][0]["predictions_file"], allow_pickle=False) as saved:
                norm = checkpoint["normalization"]
                normalized = (saved["sensors"] - np.asarray(norm["mean"])) / np.asarray(norm["std"])
                with torch.no_grad():
                    predicted, hidden = network(torch.tensor(normalized[None], dtype=torch.float32))
                self.assertEqual(hidden.shape, (1, 32))
                np.testing.assert_allclose(saved["predictions"], predicted.numpy()[0], rtol=1e-5, atol=1e-6)

    def test_64_neuron_constructor_has_expected_parameter_and_state_dimensions(self):
        try:
            network = BroadPredictor(hidden_size=64).eval()
        except TypeError as error:
            self.fail(f"Predictor must accept an explicit hidden_size: {error}")
        self.assertEqual(sum(parameter.numel() for parameter in network.parameters()), 19204)
        with torch.no_grad():
            output, hidden = network(torch.zeros(2, 5, 9))
        self.assertEqual(output.shape, (2, 5, 4))
        self.assertEqual(hidden.shape, (2, 64))
        torch.testing.assert_close(torch.linalg.vector_norm(output, dim=-1), torch.ones(2, 5), atol=1e-6, rtol=0)

    def test_invalid_widths_are_rejected_before_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared = workspace / "prepared"
            prepared_fixture(prepared)
            for width in ("0", "-1", "1.5"):
                with self.subTest(width=width):
                    output = workspace / width
                    result = subprocess.run(
                        [PYTHON, "-m", "model.broad_train", "--prepared-dir", str(prepared),
                         "--output-dir", str(output), "--hidden-size", width],
                        cwd=ROOT, capture_output=True, text=True, timeout=10,
                    )
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("hidden-size", result.stderr.lower())
                    self.assertFalse(output.exists() and any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
