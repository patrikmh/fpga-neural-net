"""Train the existing CfC network on chronological virtual-body sensor sequences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.broad_network import BroadPredictor
from model.broad_train import positive_integer, quaternion_loss, sha256
from model.broad_train_stateful import current_provenance, positive_rate
from model.orientation_generate import load_sequences


ROOT = Path(__file__).resolve().parents[1]


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    manifest_hash = sha256(args.dataset_dir / 'manifest.json')
    training = load_sequences(args.dataset_dir, 'train')
    sensors = np.concatenate([item[1] for item in training]).astype(np.float64)
    mean, std = sensors.mean(axis=0), np.maximum(sensors.std(axis=0), 1e-8)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    network = BroadPredictor(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    optimizer_steps, scored = 0, 0
    for _ in range(args.epochs):
        for _, values, targets in training:
            hidden = None
            for start in range(0, len(values), args.chunk_size):
                stop = min(start + args.chunk_size, len(values))
                inputs = torch.tensor(((values[start:stop] - mean) / std)[None], dtype=torch.float32)
                target = torch.tensor(targets[None, start:stop], dtype=torch.float32)
                mask = torch.tensor((np.arange(start, stop) >= args.warmup)[None])
                if bool(mask.any()):
                    prediction, hidden = network(inputs, hidden)
                    loss = quaternion_loss(prediction, target, mask)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
                    optimizer.step()
                    optimizer_steps += 1
                    scored += int(mask.sum())
                else:
                    with torch.no_grad():
                        _, hidden = network(inputs, hidden)
                hidden = hidden.detach()
    recipe = {name: sha256(ROOT / name) for name in (
        'model/orientation_generate.py', 'model/orientation_train.py',
        'model/orientation_evaluate.py', 'host/orientation_sim.py')}
    config = {'input_size': 9, 'hidden_size': args.hidden_size, 'output_size': 4,
              'mode': 'default', 'backbone_layers': 0, 'dt': 1., 'warmup': args.warmup,
              'epochs': args.epochs, 'chunk_size': args.chunk_size, 'seed': args.seed,
              'learning_rate': args.learning_rate, 'method': 'chronological_detached_tbptt'}
    checkpoint = {'state_dict': network.state_dict(), 'config': config,
                  'normalization': {'mean': mean.tolist(), 'std': std.tolist()},
                  'provenance': current_provenance(), 'training_recipe_sources': recipe,
                  'dataset_manifest_sha256': manifest_hash}
    if sha256(args.dataset_dir / 'manifest.json') != manifest_hash:
        raise ValueError('Dataset manifest changed during training')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / 'checkpoint.pt')
    report = {key: value for key, value in checkpoint.items() if key != 'state_dict'}
    report.update(optimizer_steps=optimizer_steps, training_samples_scored=scored)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=positive_integer, default=10)
    parser.add_argument('--hidden-size', type=positive_integer, default=96)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--chunk-size', type=positive_integer, default=256)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--learning-rate', type=positive_rate, default=.001)
    args = parser.parse_args()
    if args.seed < 0 or args.warmup < 0:
        parser.error('Seed and warmup must be nonnegative')
    try:
        train(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Orientation training failed: {error}\n')


if __name__ == '__main__':
    main()
