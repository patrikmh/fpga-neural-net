"""Export continuous validation inputs, hidden states and outputs for RTL replay."""

import argparse
import json
import math
from pathlib import Path

import torch

from model.data import GENERATOR_CONFIG, signal_trajectories
from model.integer import GoldenIntegerPredictor, quantize, sha256


def positive_integer(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("Length must be positive")
    return count


def export_vectors(export_dir, output, length):
    predictor = GoldenIntegerPredictor.from_export(export_dir)
    artifact = json.loads((export_dir / "manifest.json").read_text())
    data_path = Path(__file__).with_name("data.py")
    if artifact["sources"].get("model/data.py") != sha256(data_path):
        raise ValueError("Dataset source SHA256 identity mismatch")
    checkpoint = torch.load(export_dir / "float_checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["normalization"] != artifact["normalization"] or checkpoint["config"] != artifact["config"]:
        raise ValueError("Manifest/checkpoint configuration mismatch")
    if artifact["config"]["generator"] != GENERATOR_CONFIG:
        raise ValueError("Generator configuration mismatch")
    mean, std = (artifact["normalization"][key] for key in ("mean", "std"))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid checkpoint normalization")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    seed = artifact["config"]["seed"]
    trajectory = signal_trajectories(seed, "validation", 1, length + 1)[0]
    rows = []
    for value in trajectory[:-1]:
        x = quantize((float(value) - mean) / std)
        hidden, y = predictor.step(x)
        rows.append("".join(f"{word & 65535:04x}" for word in [x] + hidden + [y]) + "\n")
    manifest = {
        "count": length, "seed": seed, "initial_hidden": [0] * 8,
        "numeric": artifact["numeric"],
        "export_manifest_sha256": sha256(export_dir / "manifest.json"),
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "split": "validation", "trajectory_id": 0,
        "generator": artifact["config"]["generator"],
        "layout": "ten signed 16-bit words, most significant first: input,h0,h1,h2,h3,h4,h5,h6,h7,output",
        "sources": {
            "model/data.py": sha256(data_path),
            "model/integer.py": sha256(Path(__file__).with_name("integer.py")),
            "model/export_vectors.py": sha256(Path(__file__)),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / "vectors.hex"
    vectors_path.write_text("".join(rows))
    manifest["vectors_sha256"] = sha256(vectors_path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "count": length}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=positive_integer, default=10000)
    args = parser.parse_args()
    try:
        export_vectors(args.export_dir, args.output_dir, args.length)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Vector export failed: {error}\n")


if __name__ == "__main__":
    main()
