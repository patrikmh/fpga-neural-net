"""Frozen-spec integer held-out evaluation with disposable seed-17 data."""

# Behaviors covered:
# 1. A frozen spec evaluates its exact held-out split without altering artifacts.
# 2. Incompatible specs and explicit size overrides fail before output artifacts.

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from ncps.torch import CfCCell

from integration.test_model_integer import half_away, hex_words, saturate
from model.data import signal_trajectories


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path)
            for path in directory.rglob("*") if path.is_file()}


class IntegerHeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.workspace = Path(temporary.name)
        cls.trained, cls.exported = cls.workspace / "trained", cls.workspace / "exported"
        commands = [
            ["model.train", "--output-dir", str(cls.trained), "--epochs", "1", "--seed", "17",
             "--train-trajectories", "2", "--validation-trajectories", "1", "--length", "64"],
            ["model.export_integer", "--checkpoint", str(cls.trained / "checkpoint.pt"),
             "--output-dir", str(cls.exported)],
        ]
        for command in commands:
            result = subprocess.run([PYTHON, "-m", *command], cwd=ROOT,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError("Real held-out fixture must build:\n" + result.stdout + result.stderr)
        artifact = json.loads((cls.exported / "manifest.json").read_text())
        cls.spec = {
            "checkpoint_sha256": sha256(cls.trained / "checkpoint.pt"), "seed": 17,
            "trajectories": 2, "length": 64, "warmup": 8,
            "generator": artifact["config"]["generator"], "float_required_improvement": 0.2,
            "integer_epsilon_normalized_mse": 1e-6,
            "sources": {
                "model/data.py": sha256(ROOT / "model/data.py"),
                "model/network.py": sha256(ROOT / "model/network.py"),
                "ncps/torch/cfc_cell.py": sha256(Path(inspect.getfile(CfCCell))),
            },
        }

    def evaluate(self, spec_path, output, *extra):
        return subprocess.run(
            [PYTHON, "-m", "model.evaluate_integer", "--export-dir", str(self.exported),
             "--spec", str(spec_path), "--output-dir", str(output), *extra],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )

    def test_frozen_spec_evaluates_exact_heldout_data_without_changing_export(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            spec_path = workspace / "spec.json"
            spec_path.write_text(json.dumps(self.spec))
            output = workspace / "evaluated"
            before = artifact_hashes(self.exported)
            result = self.evaluate(spec_path, output)
            self.assertEqual(result.returncode, 0,
                             "Frozen-spec integer CLI must succeed:\n" + result.stdout + result.stderr)
            self.assertEqual(artifact_hashes(self.exported), before)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["split"], "test")
            self.assertEqual(report["spec"], self.spec)
            self.assertEqual(report["spec_sha256"], sha256(spec_path))
            self.assertEqual(report["checkpoint_sha256"], self.spec["checkpoint_sha256"])
            self.assertEqual(report["warmup"], 8)
            self.assertEqual(report["integer_epsilon_normalized_mse"], 1e-6)
            artifact = json.loads((self.exported / "manifest.json").read_text())
            mean, std = (artifact["normalization"][key] for key in ("mean", "std"))
            weights = hex_words(self.exported / "weights.hex")
            with np.load(output / "test_predictions.npz", allow_pickle=False) as saved:
                trajectories = saved["trajectories"]
                np.testing.assert_array_equal(trajectories, signal_trajectories(17, "test", 2, 64))
                self.assertEqual(saved["hidden"].shape, (2, 63, 8))
                for name in ("quantized_inputs", "hidden", "integer_outputs"):
                    self.assertEqual(saved[name].dtype, np.dtype("int16"))
                for name in ("integer_outputs", "integer_predictions", "float_predictions", "targets"):
                    self.assertEqual(saved[name].shape, (2, 63))
                np.testing.assert_array_equal(saved["targets"], trajectories[:, 1:])
                unbounded = [[half_away((float(value) - mean) / std * 8192) for value in row[:-1]]
                             for row in trajectories]
                np.testing.assert_array_equal(saved["quantized_inputs"],
                                               [[saturate(value) for value in row] for row in unbounded])
                np.testing.assert_allclose(saved["integer_predictions"],
                                           saved["integer_outputs"].astype(np.float64) / 8192 * std + mean,
                                           rtol=1e-6, atol=1e-7)
                self.assertEqual(report["input_clipping_count"],
                                 sum(not -32768 <= value <= 32767 for row in unbounded for value in row))
                output_clips = 0
                for hidden in saved["hidden"].reshape(-1, 8):
                    accumulator = (weights[328] << 13) + sum(int(hidden[i]) * weights[320 + i] for i in range(8))
                    output_clips += not -32768 <= half_away(accumulator / 8192) <= 32767
                self.assertEqual(report["output_clipping_count"], output_clips)
                metrics = {}
                for kind in ("float", "integer"):
                    errors = (saved[f"{kind}_predictions"][:, 8:].astype(np.float64)
                              - saved["targets"][:, 8:].astype(np.float64))
                    metrics[kind] = float(np.mean(errors ** 2) / std ** 2)
                    self.assertAlmostEqual(report[f"normalized_{kind}_mse"], metrics[kind], delta=1e-7)
                self.assertEqual(report["meets_precision_target"],
                                 metrics["integer"] <= metrics["float"] + max(0.1 * metrics["float"], 1e-6))

    def test_incompatible_specs_or_explicit_size_overrides_fail_before_artifacts(self):
        cases = ("checkpoint", "source", "generator", "trajectories", "length")
        for mismatch in cases:
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                spec = copy.deepcopy(self.spec)
                extra = []
                if mismatch == "checkpoint":
                    spec["checkpoint_sha256"] = "0" * 64
                elif mismatch == "source":
                    spec["sources"]["model/data.py"] = "0" * 64
                elif mismatch == "generator":
                    spec["generator"]["noise_std"] += 0.1
                else:
                    extra = ["--" + mismatch, "1" if mismatch == "trajectories" else "32"]
                path = workspace / "spec.json"
                path.write_text(json.dumps(spec))
                output = workspace / "evaluated"
                result = self.evaluate(path, output, *extra)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr.lower(), r"(spec|checkpoint|source|sha256|identity|generator|mismatch|override)")
                self.assertFalse(output.exists() and any(output.iterdir()),
                                 "Rejected frozen specs must not produce evaluation artifacts")


if __name__ == "__main__":
    unittest.main()
