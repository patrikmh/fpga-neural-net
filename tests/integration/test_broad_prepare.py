"""Real HDF5-to-NPZ preparation tracer with held-out files deliberately absent."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / "build/broad-analysis-env/bin/python")
REVISION = "7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad"
FIXTURE_SCRIPT = """
from pathlib import Path
import sys
import h5py
import numpy as np

directory = Path(sys.argv[1])
directory.mkdir(parents=True)
for trial in range(1, 7):
    sensors = trial * 1000 + np.arange(45, dtype=np.float64).reshape(5, 9)
    quaternion = np.array([[2,0,0,0], [0,3,0,0], [np.nan,0,0,0],
                           [1,2,2,0], [0,0,0,0]], dtype=np.float64)
    with h5py.File(directory / f'trial{trial:02d}.hdf5', 'w') as output:
        output.attrs['sampling_rate'] = 200.0
        output['imu_acc'] = sensors[:, :3]
        output['imu_gyr'] = sensors[:, 3:6]
        output['imu_mag'] = sensors[:, 6:9]
        output['opt_quat'] = quaternion
        output['movement'] = np.array([False, True, True, False, True])
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BroadPreparationTracerTests(unittest.TestCase):
    def test_preparation_preserves_train_validation_sensor_order_and_masks_without_reading_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            data = workspace / "data"
            fixture = subprocess.run(
                [PYTHON, "-c", FIXTURE_SCRIPT, str(data / "hdf5")],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(fixture.returncode, 0,
                             "Real HDF5 fixtures must be created:\n" + fixture.stdout + fixture.stderr)
            trials = []
            for trial in range(1, 11):
                filename = f"hdf5/trial{trial:02d}.hdf5"
                split = "train" if trial <= 4 else "validation" if trial <= 6 else "test"
                trials.append({"id": trial, "file": filename, "rate_hz": 200.0, "split": split,
                               "sha256": sha256(data / filename) if trial <= 6 else "f" * 64})
            spec = {"source_revision": REVISION, "trials": trials}
            spec_path = workspace / "spec.json"
            spec_path.write_text(json.dumps(spec))
            output = workspace / "prepared"
            result = subprocess.run(
                [PYTHON, "-m", "experiments.broad_prepare", "--data-dir", str(data),
                 "--spec", str(spec_path), "--output-dir", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0,
                             "BROAD preparation CLI must complete:\n" + result.stdout + result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["spec"], spec)
            self.assertEqual(manifest["spec_sha256"], sha256(spec_path))
            self.assertEqual(manifest["inputs"], {trial["file"]: trial["sha256"] for trial in trials[:6]})
            expected_outputs = {f"{trial['split']}/{trial['id']:02d}.npz" for trial in trials[:6]}
            self.assertEqual(set(manifest["outputs"]), expected_outputs)
            self.assertEqual({str(path.relative_to(output)) for path in output.rglob("*.npz")}, expected_outputs)
            expected_quaternion = np.array([[1,0,0,0], [0,1,0,0], [1,0,0,0],
                                            [1/3,2/3,2/3,0], [1,0,0,0]], dtype=np.float64)
            for trial in trials[:6]:
                with self.subTest(trial=trial["id"]):
                    filename = f"{trial['split']}/{trial['id']:02d}.npz"
                    self.assertEqual(manifest["outputs"][filename], sha256(output / filename))
                    with np.load(output / filename, allow_pickle=False) as prepared:
                        np.testing.assert_array_equal(prepared["sensors"],
                                                       trial["id"] * 1000 + np.arange(45).reshape(5, 9))
                        self.assertEqual(prepared["sensors"].shape, (5, 9))
                        self.assertEqual(prepared["quaternion"].shape, (5, 4))
                        np.testing.assert_allclose(prepared["quaternion"], expected_quaternion, rtol=1e-7, atol=1e-8)
                        np.testing.assert_array_equal(prepared["movement"], [False, True, True, False, True])
                        np.testing.assert_array_equal(prepared["reference_valid"], [True, True, False, True, False])
                        np.testing.assert_array_equal(prepared["sample_index"], np.arange(5))
                        self.assertTrue(np.isfinite(prepared["quaternion"]).all())
            for trial in trials[6:]:
                self.assertFalse((data / trial["file"]).exists(), "Held-out input files must remain absent")


if __name__ == "__main__":
    unittest.main()
