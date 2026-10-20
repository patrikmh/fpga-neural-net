"""One real BROAD integer pipeline tracer with independent scalar recurrence."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from integration.test_broad_train import prepared_fixture
from integration.test_model_integer import half_away, hex_words, saturate, shift


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_broad_step(inputs, old, weights, tanh, sigmoid):
    size = len(old)
    row_width = 10 + size
    heads = []
    clamps = state_clips = raw_clips = 0
    for head in range(4):
        rows = []
        for row in range(size):
            offset = (head * size + row) * row_width
            accumulator = weights[offset + 9 + size] << 13
            accumulator += sum((int(inputs[column]) * weights[offset + column]) << 2 for column in range(9))
            accumulator += sum(old[column] * weights[offset + 9 + column] for column in range(size))
            rows.append(accumulator)
        heads.append(rows)
    def lookup(table, accumulator):
        nonlocal clamps
        index = shift(accumulator, 18) + 2048
        clamps += not 0 <= index <= 4095
        return table[min(4095, max(0, index))]
    hidden = []
    for neuron in range(size):
        f1 = lookup(tanh, heads[0][neuron])
        f2 = lookup(tanh, heads[1][neuron])
        gate = lookup(sigmoid, heads[2][neuron] + heads[3][neuron])
        value = f1 + shift((f2-f1) * gate, 13)
        state_clips += value != saturate(value)
        hidden.append(saturate(value))
    outputs = []
    base = 4 * size * row_width
    for row in range(4):
        offset = base + row * (size + 1)
        accumulator = (weights[offset + size] << 13) + sum(hidden[column] * weights[offset + column] for column in range(size))
        value = shift(accumulator, 13)
        raw_clips += value != saturate(value)
        outputs.append(saturate(value))
    return hidden, outputs, clamps, state_clips, raw_clips


def full_rmse(prediction, target, mask):
    p, q = prediction.astype(np.float64), target.astype(np.float64)
    p /= np.linalg.norm(p,axis=1,keepdims=True)
    q /= np.linalg.norm(q,axis=1,keepdims=True)
    errors = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(p*q,axis=1)),0,1)))
    return float(np.sqrt(np.mean(errors[mask]**2)))


class BroadIntegerTracerTests(unittest.TestCase):
    def run_cli(self, *arguments):
        result = subprocess.run([PYTHON,"-m",*arguments],cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,"Actual BROAD integer pipeline must complete:\n"+result.stdout+result.stderr)

    def test_q11_inputs_and_q13_recurrent_outputs_match_scalar_export_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prepared, trained, exported, evaluated = (workspace/name for name in ("prepared","trained","exported","evaluated"))
            prepared_fixture(prepared)
            with np.load(prepared/"train/01.npz",allow_pickle=False) as source:
                validation_sensors = source["sensors"].copy()+.01
            path = prepared/"validation/03.npz"
            with np.load(path,allow_pickle=False) as source:
                validation = {name:source[name].copy() for name in source.files}
            validation["sensors"] = validation_sensors
            np.savez(path,**validation)
            manifest_path = prepared/"manifest.json"
            prepared_manifest = json.loads(manifest_path.read_text())
            prepared_manifest["outputs"]["validation/03.npz"] = sha256(path)
            manifest_path.write_text(json.dumps(prepared_manifest))
            self.run_cli("model.broad_train","--prepared-dir",str(prepared),"--output-dir",str(trained),
                         "--hidden-size","3","--epochs","1","--updates-per-epoch","2","--batch-size","2",
                         "--window","16","--warmup","2","--validation-every","1","--seed","17")
            checkpoint_path = trained/"checkpoint.pt"
            self.run_cli("model.broad_export_integer","--checkpoint",str(checkpoint_path),"--output-dir",str(exported))
            self.run_cli("model.broad_evaluate_integer","--export-dir",str(exported),"--prepared-dir",str(prepared),
                         "--output-dir",str(evaluated))

            from model.broad_integer import GoldenBroadPredictor
            from model.broad_network import BroadPredictor

            checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
            manifest = json.loads((exported/"manifest.json").read_text())
            expected_numeric = {"word_bits":16,"input_fractional_bits":11,"state_fractional_bits":13,
                                "weight_fractional_bits":13,"output_fractional_bits":13,"accumulator_bits":48,
                                "accumulator_fractional_bits":26,"rounding":"half_away_from_zero",
                                "activation_entries":4096,"activation_fractional_bits":8,"activation_zero_index":2048}
            for key,value in expected_numeric.items():
                self.assertEqual(manifest["numeric"][key],value)
            self.assertEqual(manifest["checkpoint_sha256"],sha256(checkpoint_path))
            self.assertEqual(manifest["config"],checkpoint["config"])
            self.assertEqual(manifest["normalization"],checkpoint["normalization"])
            for filename in ("weights.hex","tanh.hex","sigmoid.hex","float_checkpoint.pt"):
                self.assertEqual(manifest["files"][filename],sha256(exported/filename))
            for filename in ("model/broad_integer.py","model/broad_export_integer.py"):
                self.assertEqual(manifest["sources"][filename],sha256(ROOT/filename))
            weights,tanh,sigmoid = (hex_words(exported/name) for name in ("weights.hex","tanh.hex","sigmoid.hex"))
            self.assertEqual((len(weights),len(tanh),len(sigmoid)),(172,4096,4096))
            ordered = []
            state = checkpoint["state_dict"]
            for head in ("ff1","ff2","time_a","time_b"):
                for row in range(3):
                    ordered.extend(state[f"cell.{head}.weight"][row].tolist())
                    ordered.append(float(state[f"cell.{head}.bias"][row]))
            for row in range(4):
                ordered.extend(state["readout.weight"][row].tolist())
                ordered.append(float(state["readout.bias"][row]))
            self.assertEqual(weights,[half_away(value*8192) for value in ordered])
            for index in (0,1024,2047,2048,2049,3072,4095):
                x=(index-2048)/256
                self.assertEqual(tanh[index],half_away(math.tanh(x)*8192))
                self.assertEqual(sigmoid[index],half_away(8192/(1+math.exp(-x))))
            report = json.loads((evaluated/"report.json").read_text())
            self.assertEqual(report["split"],"validation")
            self.assertEqual(report["acceptance"]["max_full_rmse_increase_deg"],.25)
            self.assertEqual(len(report["trials"]),1)
            entry = report["trials"][0]
            self.assertEqual(entry["trial_id"],3)
            mean,std = (np.asarray(checkpoint["normalization"][key],dtype=np.float64) for key in ("mean","std"))
            with np.load(evaluated/entry["predictions_file"],allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["sensors"],validation["sensors"])
                np.testing.assert_array_equal(saved["targets"],validation["quaternion"])
                for key in ("movement","reference_valid","sample_index"):
                    np.testing.assert_array_equal(saved[key],validation[key])
                unbounded = [[half_away(value*2048) for value in row] for row in (validation_sensors.astype(np.float64)-mean)/std]
                quantized = np.array([[saturate(value) for value in row] for row in unbounded],dtype=np.int16)
                np.testing.assert_array_equal(saved["quantized_inputs"],quantized)
                self.assertTrue((quantized<0).any(),"Fixture must exercise signed input multiplication")
                self.assertEqual(saved["hidden"].shape,(24,3))
                self.assertEqual(saved["raw_outputs"].shape,(24,4))
                for key in ("quantized_inputs","hidden","raw_outputs"):
                    self.assertEqual(saved[key].dtype,np.dtype("int16"))
                predictor = GoldenBroadPredictor.from_export(exported)
                counters = {"lut_clamp_count":0,"state_clipping_count":0,"raw_clipping_count":0}
                expected_rows = []
                old = [0]*3
                predictor.reset()
                for index,inputs in enumerate(quantized.tolist()):
                    old,raw,clamps,state_clips,raw_clips = independent_broad_step(inputs,old,weights,tanh,sigmoid)
                    self.assertEqual(predictor.step(inputs),(old,raw))
                    np.testing.assert_array_equal(saved["hidden"][index],old)
                    np.testing.assert_array_equal(saved["raw_outputs"][index],raw)
                    counters["lut_clamp_count"]+=clamps
                    counters["state_clipping_count"]+=state_clips
                    counters["raw_clipping_count"]+=raw_clips
                    expected_rows.append("".join(f"{value&65535:04x}" for value in inputs+old+raw))
                self.assertTrue((saved["hidden"][1:]!=0).any(),"Fixture must exercise nonzero recurrent state")
                self.assertEqual((evaluated/entry["vectors_file"]).read_text().splitlines(),expected_rows)
                predictor.reset()
                first_h,first_raw,*_ = independent_broad_step(quantized[0].tolist(),[0]*3,weights,tanh,sigmoid)
                self.assertEqual(predictor.step(quantized[0].tolist()),(first_h,first_raw))
                raw = saved["raw_outputs"].astype(np.float64)/8192
                norm = np.linalg.norm(raw,axis=1,keepdims=True)
                zero = norm[:,0]==0
                integer_quaternions = raw/np.where(norm>0,norm,1)
                integer_quaternions[zero]=[1,0,0,0]
                np.testing.assert_allclose(saved["integer_predictions"],integer_quaternions,rtol=1e-6,atol=1e-7)
                floating = BroadPredictor(hidden_size=3).eval()
                floating.load_state_dict(state)
                with torch.no_grad():
                    predicted,_ = floating(torch.tensor(((validation_sensors.astype(np.float64)-mean)/std)[None],dtype=torch.float32))
                np.testing.assert_allclose(saved["float_predictions"],predicted.numpy()[0],rtol=1e-5,atol=1e-6)
                mask=saved["movement"]&saved["reference_valid"]
                mask[:2]=False
                measured={kind:full_rmse(saved[f"{kind}_predictions"],saved["targets"],mask) for kind in ("float","integer")}
                for kind,value in measured.items():
                    self.assertAlmostEqual(entry[kind]["full_rmse_deg"],value,delta=1e-3)
                    self.assertAlmostEqual(report["mean_trial_metrics"][kind]["full_rmse_deg"],value,delta=1e-3)
                counters["input_clipping_count"] = sum(value!=saturate(value) for row in unbounded for value in row)
                counters["zero_quaternion_count"] = int(zero.sum())
                for key,value in counters.items():
                    self.assertEqual(report[key],value)
                precision_pass = measured["integer"]<=measured["float"]+.25
                precision_pass &= all(counters[key]==0 for key in ("input_clipping_count","state_clipping_count",
                                                                   "raw_clipping_count","zero_quaternion_count"))
                self.assertEqual(report["meets_precision_target"],precision_pass)


# Additional numeric guards: signed Q11 boundaries, invalid export rejection,
# and zero-quaternion accounting across scored and unscored timesteps.
class BroadIntegerGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.workspace = Path(temporary.name)
        cls.prepared = cls.workspace / "prepared"
        prepared_fixture(cls.prepared)
        with np.load(cls.prepared / "train/01.npz", allow_pickle=False) as source:
            sensors = source["sensors"].copy() + .01
        validation_path = cls.prepared / "validation/03.npz"
        with np.load(validation_path, allow_pickle=False) as source:
            validation = {name: source[name].copy() for name in source.files}
        validation["sensors"] = sensors
        np.savez(validation_path, **validation)
        manifest_path = cls.prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["validation/03.npz"] = sha256(validation_path)
        manifest_path.write_text(json.dumps(manifest))
        cls.checkpoint = cls.workspace / "trained/checkpoint.pt"
        cls.exported = cls.workspace / "exported"
        commands = [
            ["model.broad_train", "--prepared-dir", str(cls.prepared), "--output-dir", str(cls.checkpoint.parent),
             "--hidden-size", "3", "--epochs", "1", "--updates-per-epoch", "2", "--batch-size", "2",
             "--window", "16", "--warmup", "2", "--validation-every", "1", "--seed", "17"],
            ["model.broad_export_integer", "--checkpoint", str(cls.checkpoint), "--output-dir", str(cls.exported)],
        ]
        for command in commands:
            result = subprocess.run([PYTHON, "-m", *command], cwd=ROOT, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise AssertionError("Actual numeric guard fixture must build:\n" + result.stdout + result.stderr)

    def run_cli(self, *arguments):
        return subprocess.run([PYTHON, "-m", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_signed_q11_midpoints_and_signed16_clipping_are_counted_exactly(self):
        from model.broad_integer import quantize_inputs

        values = np.array([0, .5, -.5, 1.5, -1.5, 32767, -32768, -32767.5, 32767.5, -32768.5]) / 2048
        quantized, clipped = quantize_inputs(values)
        np.testing.assert_array_equal(quantized, [0, 1, -1, 2, -2, 32767, -32768, -32768, 32767, -32768])
        self.assertEqual(quantized.dtype, np.dtype("int16"))
        self.assertEqual(clipped, 2)

    def test_invalid_coefficients_or_export_identities_fail_before_missing_prepared_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            checkpoint["state_dict"]["readout.bias"][0] = 4.0
            overflow_path = workspace / "overflow.pt"
            torch.save(checkpoint, overflow_path)
            overflow_output = workspace / "overflow-export"
            result = self.run_cli("model.broad_export_integer", "--checkpoint", str(overflow_path),
                                  "--output-dir", str(overflow_output))
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertRegex(result.stderr.lower(), r"(coefficient|q13|overflow)")
            self.assertFalse(overflow_output.exists() and any(overflow_output.iterdir()))
            prepared = workspace / "missing-samples"
            shutil.copytree(self.prepared, prepared)
            for path in prepared.rglob("*.npz"):
                path.unlink()
            for mutation in ("coefficient", "hash", "manifest"):
                with self.subTest(mutation=mutation):
                    exported = workspace / mutation
                    shutil.copytree(self.exported, exported)
                    manifest_path = exported / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    if mutation == "coefficient":
                        weights = exported / "weights.hex"
                        words = weights.read_text()
                        weights.write_text("zzzz" + words[4:])
                        manifest["files"]["weights.hex"] = sha256(weights)
                    elif mutation == "hash":
                        manifest["files"]["weights.hex"] = "0" * 64
                    else:
                        manifest["numeric"]["input_fractional_bits"] = 12
                    manifest_path.write_text(json.dumps(manifest))
                    output = workspace / (mutation + "-evaluation")
                    result = self.run_cli("model.broad_evaluate_integer", "--export-dir", str(exported),
                                          "--prepared-dir", str(prepared), "--output-dir", str(output))
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stderr.lower(), r"(hex|artifact|sha256|numeric|identity)")
                    self.assertNotIn("no such file", result.stderr.lower())
                    self.assertFalse(output.exists() and any(output.iterdir()))

    def test_quantized_zero_readout_counts_every_timestep_and_rejects_precision_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            checkpoint["state_dict"]["readout.weight"].zero_()
            checkpoint["state_dict"]["readout.bias"].fill_(1e-8)
            checkpoint_path = workspace / "tiny-readout.pt"
            torch.save(checkpoint, checkpoint_path)
            exported, output = workspace / "exported", workspace / "evaluated"
            for command in (
                ("model.broad_export_integer", "--checkpoint", str(checkpoint_path), "--output-dir", str(exported)),
                ("model.broad_evaluate_integer", "--export-dir", str(exported), "--prepared-dir", str(self.prepared),
                 "--output-dir", str(output)),
            ):
                result = self.run_cli(*command)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["meets_precision_target"])
            self.assertEqual(report["zero_quaternion_count"], 24)
            for counter in ("input_clipping_count", "state_clipping_count", "raw_clipping_count"):
                self.assertEqual(report[counter], 0)
            trial = report["trials"][0]
            with np.load(output / trial["predictions_file"], allow_pickle=False) as saved:
                self.assertEqual(saved["integer_predictions"].shape, (24, 4))
                np.testing.assert_array_equal(saved["raw_outputs"], np.zeros((24, 4), dtype=np.int16))
                np.testing.assert_array_equal(saved["integer_predictions"], np.tile([1, 0, 0, 0], (24, 1)))
                np.testing.assert_allclose(saved["float_predictions"], np.full((24, 4), .5), atol=1e-7)
                mask = saved["movement"] & saved["reference_valid"]
                mask[:2] = False
                self.assertLess(int(mask.sum()), report["zero_quaternion_count"])
                self.assertEqual(len(saved["sample_index"]), 24)


if __name__=="__main__":
    unittest.main()
