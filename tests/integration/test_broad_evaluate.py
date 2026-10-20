"""Frozen BROAD evaluation on synthetic held-out fixtures, never project data."""

import copy
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

from integration.test_broad_prepare import FIXTURE_SCRIPT
from model.broad_network import BroadPredictor


ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv/bin/python")
HDF_PYTHON = str(ROOT / "build/broad-analysis-env/bin/python")
TRAINING_HASH = "a" * 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(predicted, target, mask):
    p = predicted.astype(np.float64)
    q = target.astype(np.float64)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w = p[:,0]*q[:,0] + np.sum(p[:,1:]*q[:,1:],axis=1)
    v = -p[:,:1]*q[:,1:] + q[:,:1]*p[:,1:] - np.cross(p[:,1:],q[:,1:])
    heading_norm = np.sqrt(w*w + v[:,2]*v[:,2])
    full = np.degrees(2*np.arccos(np.clip(np.abs(w),0,1)))
    inclination = np.degrees(2*np.arccos(np.clip(heading_norm,0,1)))
    heading_mask = mask & (heading_norm > 1e-12)
    heading = np.degrees(2*np.arccos(np.clip(np.abs(w[heading_mask])/heading_norm[heading_mask],0,1)))
    return {"full_rmse_deg":float(np.sqrt(np.mean(full[mask]**2))),
            "inclination_rmse_deg":float(np.sqrt(np.mean(inclination[mask]**2))),
            "heading_rmse_deg":float(np.sqrt(np.mean(heading**2))) if len(heading) else None}


def independent_up(quaternion):
    q = quaternion / np.linalg.norm(quaternion,axis=1,keepdims=True)
    w,x,y,z = q.T
    return np.column_stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))


def independent_baselines(sensors, dt, tau):
    measured = sensors[:,:3].astype(np.float64)
    measured /= np.linalg.norm(measured,axis=1,keepdims=True)
    gyro, fused = np.empty_like(measured), np.empty_like(measured)
    gyro[0] = fused[0] = measured[0]
    alpha = 1-np.exp(-dt/tau)
    for step in range(1,len(sensors)):
        omega = -(sensors[step-1,3:6]+sensors[step,3:6]).astype(np.float64)/2
        speed = np.linalg.norm(omega)
        def propagated(up):
            if speed < 1e-12:
                return up.copy()
            axis = omega/speed
            angle = speed*dt
            return up*np.cos(angle)+np.cross(axis,up)*np.sin(angle)+axis*np.dot(axis,up)*(1-np.cos(angle))
        gyro[step] = propagated(gyro[step-1])
        gyro[step] /= np.linalg.norm(gyro[step])
        fused[step] = (1-alpha)*propagated(fused[step-1])+alpha*measured[step]
        fused[step] /= np.linalg.norm(fused[step])
    return measured,gyro,fused


class BroadEvaluationTests(unittest.TestCase):
    def fixture(self, workspace):
        torch.manual_seed(17)
        network = BroadPredictor()
        sources = {f"model/{name}":digest(ROOT/"model"/name)
                   for name in ("broad_train.py","broad_network.py")}
        sources["ncps/torch/cfc_cell.py"] = digest(Path(inspect.getfile(CfCCell)))
        versions = {"python":sys.version.split()[0],"torch":str(torch.__version__),
                    "numpy":np.__version__,"ncps":importlib.metadata.version("ncps")}
        checkpoint = {"state_dict":network.state_dict(),
                      "normalization":{"mean":[0.]*9,"std":[1.]*9},
                      "config":{"seed":17,"warmup":2,"input_size":9,"hidden_size":8,"output_size":4,
                                "mode":"default","backbone_layers":0,"dt":1.},
                      "prepared_manifest_sha256":TRAINING_HASH,
                      "provenance":{"sources":sources,"versions":versions},"best_epoch":1}
        checkpoint_path = workspace/"selected.pt"
        torch.save(checkpoint,checkpoint_path)
        frozen_sources = dict(sources)
        for name in ("model/broad_evaluate.py","experiments/broad_prepare.py"):
            frozen_sources[name] = digest(ROOT/name) if (ROOT/name).exists() else "0"*64
        spec = {"source_revision":"7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad",
                "checkpoint_sha256":digest(checkpoint_path),"training_manifest_sha256":TRAINING_HASH,
                "selection_sha256":"b"*64,"sources":frozen_sources,"versions":versions,
                "warmup":2,"baseline_tau_s":.5,"heading_epsilon":1e-12,"trials":[]}
        prepared = workspace/"prepared"
        inputs,outputs = {},{}
        for trial,length in ((7,12),(8,17)):
            t = np.arange(length,dtype=np.float64)
            sensors = np.column_stack((.1*np.sin(t),.1*np.cos(t),np.full(length,9.81),
                                       .02*np.sin(t/3),.01*np.cos(t/2),np.full(length,.03),
                                       np.ones(length),np.zeros((length,2)))).astype(np.float32)
            angle = .03*t+trial*.01
            target = np.column_stack((np.cos(angle/2),np.zeros((length,2)),np.sin(angle/2))).astype(np.float32)
            valid = np.ones(length,dtype=bool)
            valid[5] = False
            target[5] = [1,0,0,0]
            movement = np.ones(length,dtype=bool)
            movement[0] = False
            filename = f"test/{trial:02d}.npz"
            path = prepared/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,sensors=sensors,quaternion=target,movement=movement,
                     reference_valid=valid,sample_index=np.arange(length))
            raw = f"hdf5/trial{trial:02d}.hdf5"
            inputs[raw] = hashlib.sha256(raw.encode()).hexdigest()
            spec["trials"].append({"id":trial,"split":"test","file":raw,"sha256":inputs[raw],"rate_hz":100.})
            outputs[filename] = digest(path)
        spec_path = workspace/"spec.json"
        spec_path.write_text(json.dumps(spec))
        manifest = {"spec":spec,"spec_sha256":digest(spec_path),"inputs":inputs,"outputs":outputs,
                    "sources":{"experiments/broad_prepare.py":frozen_sources["experiments/broad_prepare.py"]}}
        (prepared/"manifest.json").write_text(json.dumps(manifest))
        return checkpoint_path,prepared,spec_path,spec,network

    def evaluate(self, checkpoint, prepared, spec, output):
        return subprocess.run([PYTHON,"-m","model.broad_evaluate","--checkpoint",str(checkpoint),
                               "--prepared-dir",str(prepared),"--spec",str(spec),"--output-dir",str(output)],
                              cwd=ROOT,capture_output=True,text=True,timeout=30)

    def test_selected_checkpoint_evaluates_bound_test_trials_and_independent_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            checkpoint,prepared,spec_path,spec,network=self.fixture(workspace)
            output=workspace/"evaluated"
            result=self.evaluate(checkpoint,prepared,spec_path,output)
            self.assertEqual(result.returncode,0,"Actual BROAD evaluator must complete:\n"+result.stdout+result.stderr)
            report=json.loads((output/"report.json").read_text())
            for key in ("checkpoint_sha256","training_manifest_sha256","selection_sha256","sources","versions"):
                self.assertEqual(report[key],spec[key])
            self.assertEqual(report["spec"],spec)
            self.assertEqual(report["spec_sha256"],digest(spec_path))
            self.assertEqual([entry["trial_id"] for entry in report["trials"]],[7,8])
            all_cfc=[]
            all_baselines=[]
            network.eval()
            baseline_keys=("accelerometer_only","gyro_only_acc_initialization","gyro_acc_complementary")
            for entry in report["trials"]:
                with np.load(output/entry["predictions_file"],allow_pickle=False) as saved:
                    with np.load(prepared/f"test/{entry['trial_id']:02d}.npz",allow_pickle=False) as source:
                        np.testing.assert_array_equal(saved["targets"],source["quaternion"])
                        for key in ("sensors","movement","reference_valid","sample_index"):
                            np.testing.assert_array_equal(saved[key],source[key])
                    with torch.no_grad():
                        predictions,_=network(torch.tensor(saved["sensors"][None],dtype=torch.float32))
                    np.testing.assert_allclose(saved["predictions"],predictions.numpy()[0],rtol=1e-5,atol=1e-6)
                    mask=saved["movement"] & saved["reference_valid"]
                    mask[:2]=False
                    metrics=independent_metrics(saved["predictions"],saved["targets"],mask)
                    all_cfc.append(metrics)
                    for name,value in metrics.items():
                        self.assertAlmostEqual(entry["cfc"][name],value,delta=1e-3)
                    reference_up=independent_up(saved["targets"].astype(np.float64))
                    baselines={}
                    for key,array_name,up in zip(baseline_keys,("accelerometer_up","gyro_up","complementary_up"),
                                                 independent_baselines(saved["sensors"],.01,.5)):
                        np.testing.assert_allclose(saved[array_name],up,rtol=1e-6,atol=1e-7)
                        errors=np.degrees(np.arccos(np.clip(np.sum(up*reference_up,axis=1),-1,1)))
                        baselines[key]=float(np.sqrt(np.mean(errors[mask]**2)))
                        self.assertAlmostEqual(entry["baselines"][key]["incl_rmse_deg"],baselines[key],delta=1e-4)
                        self.assertNotIn("heading_rmse_deg",entry["baselines"][key])
                    all_baselines.append(baselines)
            for name in all_cfc[0]:
                self.assertAlmostEqual(report["mean_trial_metrics"]["cfc"][name],
                                       float(np.mean([metrics[name] for metrics in all_cfc])),delta=1e-3)
            for key in baseline_keys:
                self.assertAlmostEqual(report["mean_trial_metrics"]["baselines"][key]["incl_rmse_deg"],
                                       float(np.mean([metrics[key] for metrics in all_baselines])),delta=1e-4)

    def test_frozen_identity_mismatches_are_rejected_before_any_test_npz_access(self):
        for mismatch in ("checkpoint","training_manifest","source"):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as directory:
                workspace=Path(directory)
                checkpoint,prepared,spec_path,spec,_=self.fixture(workspace)
                for path in prepared.rglob("*.npz"):
                    path.unlink()
                if mismatch=="source":
                    spec["sources"]["model/broad_network.py"]="0"*64
                else:
                    spec[f"{mismatch}_sha256"]="0"*64
                spec_path.write_text(json.dumps(spec))
                manifest=json.loads((prepared/"manifest.json").read_text())
                manifest.update(spec=spec,spec_sha256=digest(spec_path))
                (prepared/"manifest.json").write_text(json.dumps(manifest))
                output=workspace/"evaluated"
                result=self.evaluate(checkpoint,prepared,spec_path,output)
                self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertRegex(result.stderr.lower(),r"(identity|mismatch|sha256|source|binding)")
                self.assertNotIn("no such file",result.stderr.lower())
                self.assertFalse(output.exists() and any(output.iterdir()))

    def test_explicit_test_conversion_reads_only_declared_test_hdf5_and_binds_frozen_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            _,_,_,spec,_=self.fixture(workspace)
            raw=workspace/"raw"
            fixture=subprocess.run([HDF_PYTHON,"-c",FIXTURE_SCRIPT,str(raw/"hdf5")],
                                   cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(fixture.returncode,0,fixture.stdout+fixture.stderr)
            spec=copy.deepcopy(spec)
            spec["trials"]=[{"id":1,"split":"test","file":"hdf5/trial01.hdf5",
                              "sha256":digest(raw/"hdf5/trial01.hdf5"),"rate_hz":200.}]
            path=workspace/"frozen.json"
            path.write_text(json.dumps(spec))
            output=workspace/"converted"
            result=subprocess.run([HDF_PYTHON,"-m","experiments.broad_prepare","--data-dir",str(raw),
                                   "--spec",str(path),"--output-dir",str(output),"--split","test"],
                                  cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,"Explicit test conversion must succeed:\n"+result.stdout+result.stderr)
            manifest=json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["spec"],spec)
            self.assertEqual(manifest["spec_sha256"],digest(path))
            self.assertEqual(set(manifest["outputs"]),{"test/01.npz"})
            self.assertEqual(manifest["outputs"]["test/01.npz"],digest(output/"test/01.npz"))
            with np.load(output/"test/01.npz",allow_pickle=False) as trial:
                self.assertEqual(trial["sensors"].shape,(5,9))
                np.testing.assert_array_equal(trial["reference_valid"],[True,True,False,True,False])


if __name__=="__main__":
    unittest.main()
