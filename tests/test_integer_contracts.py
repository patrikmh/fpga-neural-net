"""Fixed-point boundary and frozen-artifact integrity contracts."""

# Behaviors covered:
# 1. Signed ties round away from zero and outputs saturate to signed 16 bits.
# 2. Recurrent neurons read old state; readout reads new state; results are copies.
# 3. Affine accumulators preserve values beyond signed 32-bit range.
# 4. Activation tables obey the narrower tanh/sigmoid Q13 ranges.
# 5. Artifact loading rejects weight corruption and stale integer source identity.
# 6. Export rejects unsupported checkpoint configurations and coefficient overflow.
# 7. Evaluation rejects changed data/network/ncps source identities.

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch

from model.integer import GoldenIntegerPredictor, round_shift, saturate


ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")


def rounded(value):
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def tables():
    grid = [(index - 2048) / 256 for index in range(4096)]
    return ([rounded(math.tanh(x) * 8192) for x in grid],
            [rounded(8192 / (1 + math.exp(-x))) for x in grid])


class IntegerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.workspace = Path(temporary.name)
        cls.checkpoint_path = cls.workspace / "trained/checkpoint.pt"
        cls.exported = cls.workspace / "exported"
        commands = [
            ["model.train", "--output-dir", str(cls.checkpoint_path.parent), "--seed", "17",
             "--epochs", "1", "--train-trajectories", "2", "--validation-trajectories", "1", "--length", "64"],
            ["model.export_integer", "--checkpoint", str(cls.checkpoint_path), "--output-dir", str(cls.exported)],
        ]
        for command in commands:
            result = subprocess.run([PYTHON, "-m", *command], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise AssertionError("Real integer fixture must build:\n" + result.stdout + result.stderr)

    def run_cli(self, *arguments):
        return subprocess.run([PYTHON, "-m", *arguments], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)

    def test_signed_rounding_ties_and_saturation_boundaries(self):
        for value, expected in ((4096, 1), (-4096, -1), (4095, 0), (-4095, 0),
                                (12288, 2), (-12288, -2)):
            with self.subTest(value=value):
                self.assertEqual(round_shift(value, 13), expected)
        for value, expected in ((-32769, -32768), (-32768, -32768), (32767, 32767), (32768, 32767)):
            with self.subTest(saturation=value):
                self.assertEqual(saturate(value), expected)

    def test_neurons_use_old_state_readout_uses_new_state_and_returned_state_is_a_copy(self):
        tanh, sigmoid = tables()
        weights = [0] * 329
        for head in (0, 80):
            weights[head] = 8192
            weights[head + 11] = 8192
        weights[321] = 8192
        predictor = GoldenIntegerPredictor(weights, tanh, sigmoid)

        first, first_output = predictor.step(8192)
        self.assertEqual((first, first_output), ([6239, 0, 0, 0, 0, 0, 0, 0], 0))
        first[0] = -32768
        second, second_output = predictor.step(0)
        expected = tanh[2243]
        self.assertEqual((second, second_output), ([0, expected, 0, 0, 0, 0, 0, 0], expected))

    def test_affine_accumulators_preserve_values_beyond_signed_32_bits(self):
        tanh, sigmoid = tables()
        predictor = GoldenIntegerPredictor([32767] * 329, tanh, sigmoid)
        self.assertEqual(predictor.step(32767), ([8192] * 8, 32767))
        independent_accumulator = (32767 << 13) + 32767 * 32767 + 8 * 8192 * 32767
        self.assertGreater(independent_accumulator, 2 ** 31 - 1)
        self.assertEqual(predictor.step(32767), ([8192] * 8, 32767))

    def test_activation_tables_reject_values_outside_their_q13_ranges(self):
        for table_name, bad_value in (("tanh", -8193), ("tanh", 8193), ("sigmoid", -1), ("sigmoid", 8193)):
            with self.subTest(table=table_name, value=bad_value):
                tanh, sigmoid = tables()
                (tanh if table_name == "tanh" else sigmoid)[2048] = bad_value
                with self.assertRaisesRegex(ValueError, r"(?i)(activation|tanh|sigmoid|range)"):
                    GoldenIntegerPredictor([0] * 329, tanh, sigmoid)

    def test_artifact_loading_rejects_corrupted_weights_or_stale_integer_source(self):
        for mutation in ("weight", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                exported = Path(directory) / "exported"
                shutil.copytree(self.exported, exported)
                if mutation == "weight":
                    path = exported / "weights.hex"
                    original = path.read_text()
                    path.write_text(("0001" if original[:4] != "0001" else "0002") + original[4:])
                else:
                    path = exported / "manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["sources"]["model/integer.py"] = "0" * 64
                    path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, r"(?i)(sha256|source|identity|mismatch)"):
                    GoldenIntegerPredictor.from_export(exported)

    def test_export_rejects_unsupported_checkpoint_configuration_or_coefficient_overflow(self):
        original = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")
        changes = (("mode", "no_gate"), ("dt", 0.5), ("hidden_size", 16),
                   ("backbone_layers", 1), ("input_size", 2), ("coefficient", 4.0))
        for field, value in changes:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                changed = copy.deepcopy(original)
                if field == "coefficient":
                    changed["state_dict"]["cell.ff1.weight"][0, 0] = value
                else:
                    changed["config"][field] = value
                checkpoint_path = workspace / "changed.pt"
                torch.save(changed, checkpoint_path)
                output = workspace / "exported"
                result = self.run_cli("model.export_integer", "--checkpoint", str(checkpoint_path),
                                      "--output-dir", str(output))
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr.lower(), r"(config|unsupported|coefficient|q13|fit)")
                self.assertFalse(output.exists() and any(output.iterdir()))

    def test_evaluator_rejects_data_network_or_ncps_source_identity_drift(self):
        for source in ("model/data.py", "model/network.py", "ncps/torch/cfc_cell.py"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                exported = workspace / "exported"
                shutil.copytree(self.exported, exported)
                path = exported / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["sources"][source] = "0" * 64
                path.write_text(json.dumps(manifest))
                output = workspace / "evaluated"
                result = self.run_cli("model.evaluate_integer", "--export-dir", str(exported),
                                      "--output-dir", str(output), "--trajectories", "1", "--length", "64")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stderr.lower(), r"(source|sha256|identity|mismatch)")
                self.assertFalse(output.exists() and any(output.iterdir()))


if __name__ == "__main__":
    unittest.main()
