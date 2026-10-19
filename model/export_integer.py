"""Export a frozen CfC checkpoint to self-contained signed Q13 memories."""

import argparse
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.integer import NUMERIC, half_away, sha256


def export(checkpoint_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    required_config = {"mode": "default", "dt": 1, "input_size": 1,
                       "hidden_size": 8, "backbone_layers": 0}
    for name, value in required_config.items():
        if checkpoint["config"].get(name) != value:
            raise ValueError(f"Unsupported checkpoint configuration: {name}")
    state = checkpoint["state_dict"]
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    mean, std = (checkpoint["normalization"][key] for key in ("mean", "std"))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid checkpoint normalization")
    ordered = []
    for head in ("ff1", "ff2", "time_a", "time_b"):
        for neuron in range(8):
            ordered.extend(state[f"cell.{head}.weight"][neuron].tolist())
            ordered.append(float(state[f"cell.{head}.bias"][neuron]))
    ordered.extend(state["readout.weight"][0].tolist())
    ordered.append(float(state["readout.bias"][0]))
    weights = [half_away(value * 8192) for value in ordered]
    if len(weights) != 329 or any(not -32768 <= value <= 32767 for value in weights):
        raise ValueError("Checkpoint coefficients do not fit signed Q13")
    grid = [(index - 2048) / 256 for index in range(4096)]
    tables = {
        "weights.hex": weights,
        "tanh.hex": [half_away(math.tanh(value) * 8192) for value in grid],
        "sigmoid.hex": [half_away(8192 / (1 + math.exp(-value))) for value in grid],
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in tables.items():
        (output / filename).write_text("".join(f"{value & 65535:04x}\n" for value in values))
    shutil.copyfile(checkpoint_path, output / "float_checkpoint.pt")
    sources = {
        f"model/{name}": sha256(Path(__file__).with_name(name))
        for name in ("integer.py", "export_integer.py", "data.py", "network.py")
    }
    sources["ncps/torch/cfc_cell.py"] = sha256(Path(inspect.getfile(CfCCell)))
    manifest = {
        "numeric": NUMERIC, "checkpoint_sha256": sha256(checkpoint_path),
        "normalization": checkpoint["normalization"], "config": checkpoint["config"],
        "files": {name: sha256(output / name) for name in (*tables, "float_checkpoint.pt")},
        "sources": sources,
        "versions": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                     "numpy": np.__version__, "ncps": importlib.metadata.version("ncps")},
        "rules": {
            "weights": "ff1,ff2,time_a,time_b: 8 rows of 9 weights then bias; readout 8 weights then bias",
            "affine": "Q26: bias<<13 + dot([input,old_hidden],weights); 48-bit signed accumulator",
            "activation_index": "clamp(round_shift(acc,18)+2048,0,4095)",
            "hidden": "sat16(f1+round_shift((f2-f1)*sigmoid(time_a+time_b),13)); dt=1",
            "output": "sat16(round_shift((readout_bias<<13)+dot(new_hidden,readout_weights),13))",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "coefficients": len(weights)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Integer export failed: {error}\n")


if __name__ == "__main__":
    main()
