"""Train the float CfC on CPU and retain the best validation checkpoint."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.data import GENERATOR_CONFIG, fitnormalization, signal_trajectories
from model.network import SignalPredictor


WARMUP = 8


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def raw_mse(predictions, targets):
    return float(np.mean((predictions.astype(np.float64) - targets.astype(np.float64)) ** 2))


def train(args):
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    training = signal_trajectories(args.seed, "train", args.train_trajectories, args.length)
    validation = signal_trajectories(args.seed, "validation", args.validation_trajectories, args.length)
    normalization = fitnormalization(training)
    mean, std = normalization["mean"], normalization["std"]
    config = {
        "seed": args.seed, "epochs": args.epochs, "train_trajectories": args.train_trajectories,
        "validation_trajectories": args.validation_trajectories, "length": args.length,
        "warmup": WARMUP, "normalization": normalization, "generator": GENERATOR_CONFIG,
        "input_size": 1, "hidden_size": 8, "mode": "default", "backbone_layers": 0,
        "dt": 1.0, "optimizer": "Adam", "learning_rate": 0.003,
    }
    inputs = torch.tensor((training[:, :-1, None] - mean) / std, dtype=torch.float32)
    targets = torch.tensor((training[:, 1:, None] - mean) / std, dtype=torch.float32)
    validation_inputs = torch.tensor((validation[:, :-1, None] - mean) / std, dtype=torch.float32)
    network = SignalPredictor()
    optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        network.train()
        optimizer.zero_grad()
        predictions = network(inputs)
        loss = torch.mean((predictions[:, WARMUP:] - targets[:, WARMUP:]) ** 2)
        loss.backward()
        optimizer.step()
        network.eval()
        with torch.no_grad():
            validation_predictions = network(validation_inputs).numpy()[:, WARMUP:, 0] * std + mean
            validation_mse = raw_mse(validation_predictions, validation[:, WARMUP + 1:])
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
                training_loss = float(torch.mean((network(inputs)[:, WARMUP:] - targets[:, WARMUP:]) ** 2))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs}: validation MSE {validation_mse:.8g}", file=sys.stderr)
    network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        predictions = network(validation_inputs).numpy()[:, WARMUP:, 0] * std + mean
    current = validation[:, WARMUP:-1]
    previous = validation[:, WARMUP - 1:-2]
    raw_targets = validation[:, WARMUP + 1:]
    torch.save({"state_dict": best_state, "config": config, "normalization": normalization}, output / "checkpoint.pt")
    np.savez(output / "validation_predictions.npz", trajectories=validation, inputs=current,
             previous=previous, targets=raw_targets, predictions=predictions)
    cell_source = Path(inspect.getfile(CfCCell))
    local_sources = [Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("network.py")]
    report = {
        "config": config, "parameter_count": sum(p.numel() for p in network.parameters()),
        "best_epoch": best_epoch, "training_loss": training_loss,
        "validation_mse": raw_mse(predictions, raw_targets),
        "persistence_mse": raw_mse(current, raw_targets),
        "linear_mse": raw_mse(2 * current - previous, raw_targets),
        "provenance": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "ncps": importlib.metadata.version("ncps"),
            "ncps_cell_sha256": hashlib.sha256(cell_source.read_bytes()).hexdigest(),
            "sources": {f"model/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest() for path in local_sources},
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(output), "best_epoch": best_epoch, "validation_mse": report["validation_mse"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-trajectories", type=positive_integer, default=32)
    parser.add_argument("--validation-trajectories", type=positive_integer, default=8)
    parser.add_argument("--length", type=positive_integer, default=256)
    args = parser.parse_args()
    if args.length <= WARMUP + 1:
        parser.error("length must exceed warmup + 1 (9)")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        train(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
