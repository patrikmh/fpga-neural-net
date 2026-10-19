"""Compare a frozen integer export with float on validation or a frozen test spec."""

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.data import signal_trajectories
from model.integer import GoldenIntegerPredictor, half_away, quantize, sha256
from model.network import SignalPredictor


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def evaluate(export_dir, output, trajectories_count=None, length=None, spec_path=None):
    if spec_path is not None and (trajectories_count is not None or length is not None):
        raise ValueError("Frozen spec dimensions cannot be overridden")
    predictor = GoldenIntegerPredictor.from_export(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    source_paths = {
        "model/integer.py": Path(__file__).with_name("integer.py"),
        "model/evaluate_integer.py": Path(__file__),
        "model/data.py": Path(__file__).with_name("data.py"),
        "model/network.py": Path(__file__).with_name("network.py"),
        "ncps/torch/cfc_cell.py": Path(inspect.getfile(CfCCell)),
    }
    runtime_sources = {name: sha256(path) for name, path in source_paths.items()}
    for name in ("model/data.py", "model/network.py", "ncps/torch/cfc_cell.py"):
        if manifest["sources"].get(name) != runtime_sources[name]:
            raise ValueError(f"Source SHA256 identity mismatch: {name}")
    checkpoint = torch.load(export_dir / "float_checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["normalization"] != manifest["normalization"] or checkpoint["config"] != manifest["config"]:
        raise ValueError("Manifest/checkpoint configuration mismatch")
    split = "validation"
    epsilon = 1e-6
    seed = manifest["config"]["seed"]
    warmup = manifest["config"]["warmup"]
    spec = None
    if spec_path is not None:
        spec = json.loads(spec_path.read_text())
        if spec["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
            raise ValueError("Spec/checkpoint SHA256 identity mismatch")
        expected_sources = {name: runtime_sources[name]
                            for name in ("model/data.py", "model/network.py", "ncps/torch/cfc_cell.py")}
        if spec["sources"] != expected_sources:
            raise ValueError("Spec source SHA256 identity mismatch")
        for name in ("seed", "generator", "warmup"):
            if spec[name] != manifest["config"][name]:
                raise ValueError(f"Spec/checkpoint {name} mismatch")
        for name in ("seed", "trajectories", "length", "warmup"):
            if type(spec[name]) is not int:
                raise ValueError(f"Spec {name} must be an integer")
        if spec["seed"] < 0 or spec["trajectories"] <= 0 or not 1 <= spec["warmup"] < spec["length"] - 1:
            raise ValueError("Invalid spec dimensions, seed or warmup")
        epsilon = spec["integer_epsilon_normalized_mse"]
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Spec epsilon must be finite and positive")
        split = "test"
        trajectories_count, length = spec["trajectories"], spec["length"]
    else:
        trajectories_count = 8 if trajectories_count is None else trajectories_count
        length = 1024 if length is None else length
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    if length <= warmup + 1:
        raise ValueError("Length must leave targets after warmup")
    mean, std = (manifest["normalization"][key] for key in ("mean", "std"))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid normalization")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trajectories = signal_trajectories(seed, split, trajectories_count, length)
    normalized = [[(float(value) - mean) / std for value in row[:-1]] for row in trajectories]
    input_clipping_count = sum(not -32768 <= half_away(value * 8192) <= 32767 for row in normalized for value in row)
    quantized = np.array([[quantize(value) for value in row] for row in normalized], dtype=np.int16)
    hidden = np.empty((trajectories_count, length - 1, 8), dtype=np.int16)
    outputs = np.empty((trajectories_count, length - 1), dtype=np.int16)
    output_clipping_count = 0
    for trajectory in range(trajectories_count):
        predictor.reset()
        for step, value in enumerate(quantized[trajectory]):
            hidden[trajectory, step], outputs[trajectory, step] = predictor.step(int(value))
        output_clipping_count += predictor.output_clipping_count
    integer_predictions = outputs.astype(np.float64) / 8192 * std + mean
    network = SignalPredictor().eval()
    network.load_state_dict(checkpoint["state_dict"])
    inputs = torch.tensor((trajectories[:, :-1, None] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        float_predictions = network(inputs).numpy()[:, :, 0] * std + mean
    targets = trajectories[:, 1:]
    errors = {}
    for name, values in (("float", float_predictions), ("integer", integer_predictions)):
        errors[name] = float(np.mean((values[:, warmup:].astype(np.float64) - targets[:, warmup:].astype(np.float64)) ** 2))
    normalized_errors = {name: value / std ** 2 for name, value in errors.items()}
    report = {
        "split": split, "warmup": warmup, "trajectories": trajectories_count, "length": length,
        "raw_float_mse": errors["float"], "raw_integer_mse": errors["integer"],
        "normalized_float_mse": normalized_errors["float"],
        "normalized_integer_mse": normalized_errors["integer"],
        "integer_epsilon_normalized_mse": epsilon,
        "meets_precision_target": normalized_errors["integer"] <= normalized_errors["float"] + max(0.1 * normalized_errors["float"], epsilon),
        "input_clipping_count": input_clipping_count, "output_clipping_count": output_clipping_count,
        "manifest_sha256": sha256(export_dir / "manifest.json"),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "sources": manifest["sources"], "versions": manifest["versions"],
        "runtime_sources": runtime_sources,
        "runtime_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                             "numpy": np.__version__, "ncps": importlib.metadata.version("ncps")},
    }
    if spec is not None:
        report.update(spec=spec, spec_sha256=sha256(spec_path))
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f"{split}_predictions.npz", trajectories=trajectories,
             quantized_inputs=quantized, hidden=hidden, integer_outputs=outputs,
             integer_predictions=integer_predictions, float_predictions=float_predictions, targets=targets)
    (output / "report.json").write_text(report_json)
    print(json.dumps({"output_dir": str(output), "meets_precision_target": report["meets_precision_target"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=positive_integer)
    parser.add_argument("--length", type=positive_integer)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    try:
        evaluate(args.export_dir, args.output_dir, args.trajectories, args.length, args.spec)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Integer evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
