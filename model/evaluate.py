"""Evaluate a frozen checkpoint against an identity-bound held-out test spec."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.data import GENERATOR_CONFIG, signal_trajectories
from model.network import SignalPredictor


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(trajectories, full_predictions, std, warmup, required_improvement):
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Normalization standard deviation must be finite and positive")
    if not math.isfinite(required_improvement) or not 0 <= required_improvement <= 1:
        raise ValueError("Required improvement must be between zero and one")
    if trajectories.ndim != 2 or full_predictions.shape != (trajectories.shape[0], trajectories.shape[1] - 1):
        raise ValueError("Predictions must align with next-sample trajectory targets")
    if not isinstance(warmup, int) or not 1 <= warmup < trajectories.shape[1] - 1:
        raise ValueError("Warmup must leave at least one next-sample target")
    if not np.isfinite(trajectories).all() or not np.isfinite(full_predictions).all():
        raise ValueError("Trajectories and predictions must be finite")
    targets = trajectories[:, warmup + 1:]
    current = trajectories[:, warmup:-1]
    previous = trajectories[:, warmup - 1:-2]
    candidates = {"cfc": full_predictions[:, warmup:], "persistence": current,
                  "linear": 2 * current - previous}
    errors = {name: (values.astype(np.float64) - targets.astype(np.float64)) ** 2
              for name, values in candidates.items()}
    raw = {name: float(np.mean(error)) for name, error in errors.items()}
    per_trajectory = []
    for index in range(len(trajectories)):
        trajectory_raw = {name: float(np.mean(error[index])) for name, error in errors.items()}
        per_trajectory.append({
            "raw_mse": trajectory_raw,
            "normalized_mse": {name: value / std ** 2 for name, value in trajectory_raw.items()},
        })
    return {
        "raw_mse": raw,
        "normalized_mse": {name: value / std ** 2 for name, value in raw.items()},
        "per_trajectory": per_trajectory,
        "meets_float_target": raw["cfc"] <= (1 - required_improvement) * raw["persistence"],
    }


def evaluate(checkpoint_path, spec_path, output):
    spec = json.loads(spec_path.read_text())
    checkpoint_hash = sha256(checkpoint_path)
    if spec["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint SHA256 identity mismatch")
    source_paths = {
        "model/data.py": Path(__file__).with_name("data.py"),
        "model/network.py": Path(__file__).with_name("network.py"),
        "ncps/torch/cfc_cell.py": Path(inspect.getfile(CfCCell)),
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    if spec["sources"] != source_hashes:
        raise ValueError("Source SHA256 identity mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    for key in ("seed", "warmup", "generator"):
        if spec[key] != config[key]:
            raise ValueError(f"Spec/checkpoint {key} mismatch")
    if spec["generator"] != GENERATOR_CONFIG:
        raise ValueError("Generator configuration mismatch")
    for key in ("seed", "trajectories", "length", "warmup"):
        if type(spec[key]) is not int:
            raise ValueError(f"Spec {key} must be an integer")
    if spec["seed"] < 0 or spec["trajectories"] <= 0 or not 1 <= spec["warmup"] < spec["length"] - 1:
        raise ValueError("Invalid test dimensions, seed or warmup")
    improvement = spec["float_required_improvement"]
    epsilon = spec["integer_epsilon_normalized_mse"]
    if not math.isfinite(improvement) or not 0 <= improvement <= 1:
        raise ValueError("Invalid required improvement")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Integer normalized MSE epsilon must be finite and positive")
    mean, std = (checkpoint["normalization"][key] for key in ("mean", "std"))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid checkpoint normalization")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    network = SignalPredictor().eval()
    network.load_state_dict(checkpoint["state_dict"])
    trajectories = signal_trajectories(spec["seed"], "test", spec["trajectories"], spec["length"])
    normalized = torch.tensor((trajectories[:, :-1, None] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        predictions = network(normalized).numpy()[:, :, 0] * std + mean
    warmup = spec["warmup"]
    report = metrics(trajectories, predictions, std, warmup, improvement)
    report.update(
        spec=spec, spec_sha256=sha256(spec_path), checkpoint_sha256=checkpoint_hash,
        sources=source_hashes,
        versions={"python": sys.version.split()[0], "torch": str(torch.__version__),
                  "numpy": np.__version__, "ncps": importlib.metadata.version("ncps")},
    )
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "test_predictions.npz", trajectories=trajectories,
             inputs=trajectories[:, warmup:-1], previous=trajectories[:, warmup - 1:-2],
             targets=trajectories[:, warmup + 1:], predictions=predictions[:, warmup:])
    (output / "report.json").write_text(report_json)
    print(json.dumps({"output_dir": str(output), "meets_float_target": report["meets_float_target"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args.checkpoint, args.spec, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
