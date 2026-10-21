"""Train the existing CfC hardware architecture to forecast future orientation."""

import argparse
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
import torch

from model.broad_integer import artifact_sources, validate_checkpoint
from model.broad_network import BroadPredictor
from model.broad_train import positive_integer, sha256
from model.broad_train_stateful import current_provenance, positive_rate
from model.orientation_evaluate import metrics
from model.motion_generate import load_sequences


ROOT = Path(__file__).resolve().parents[1]



def forecast_loss(raw, targets, mask):
    """Sign-invariant chordal direction loss with a small unit-norm penalty."""
    if raw.shape != targets.shape or raw.shape[-1] != 4 or mask.shape != raw.shape[:-1]:
        raise ValueError('Forecast loss requires matching quaternion arrays and frame mask')
    raw, targets = raw[mask], targets[mask]
    if not len(raw):
        raise ValueError('Forecast loss requires at least one scored frame')
    raw_norm = torch.linalg.vector_norm(raw, dim=-1)
    target_norm = torch.linalg.vector_norm(targets, dim=-1)
    if (not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(targets).all())
            or bool((raw_norm == 0).any()) or bool((target_norm == 0).any())):
        raise ValueError('Scored forecast and target quaternions must be finite and nonzero')
    alignment = torch.sum((raw / raw_norm[:, None]) * (targets / target_norm[:, None]), dim=-1)
    return (1 - alignment.abs().clamp(max=1) + .01 * (raw_norm - 1).square()).mean()


def raw_sequence(network, inputs, hidden=None):
    if hidden is None:
        hidden = inputs.new_zeros((inputs.shape[0], network.hidden_size))
    outputs = []
    for step in range(inputs.shape[1]):
        values, hidden = network.cell(inputs[:, step], hidden, 1.)
        raw = network.readout(values)
        if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(raw).all()):
            raise ValueError('Nonfinite training hidden state or quaternion')
        outputs.append(raw)
    return torch.stack(outputs, dim=1), hidden

def batches(sequences, batch_size):
    for _, group in groupby(sequences, key=lambda item: len(item[1])):
        group = list(group)
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def validation_metrics(network, sequences, batch_size, warmup):
    errors, families = [], {}
    network.eval()
    with torch.no_grad():
        for batch in batches(sorted(sequences, key=lambda item: len(item[1])), batch_size):
            inputs = torch.stack([item[1] for item in batch])
            hidden = inputs.new_zeros((len(batch), network.hidden_size))
            raw_outputs = []
            for step in range(inputs.shape[1]):
                output, hidden = network.cell(inputs[:, step], hidden, 1.)
                raw = network.readout(output)
                if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(raw).all()):
                    raise ValueError('Nonfinite validation hidden state or quaternion')
                raw_outputs.append(raw)
            raw = torch.stack(raw_outputs, dim=1).numpy().astype(np.float64)
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError('Model emitted a zero validation quaternion')
            predictions = raw / norms
            for index, (entry, _, targets) in enumerate(batch):
                target = targets.numpy().astype(np.float64)
                target /= np.linalg.norm(target, axis=-1, keepdims=True)
                angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(
                    predictions[index] * target, axis=-1)), 0, 1)))[warmup:]
                errors.extend(angle.tolist())
                families.setdefault(entry['family'], []).extend(angle.tolist())
    if not errors:
        raise ValueError('No scored validation samples remain after warmup')
    result = metrics(errors)
    result['families'] = {name: metrics(values) for name, values in families.items()}
    return result


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    manifest_hash = sha256(args.dataset_dir / 'manifest.json')
    dataset_manifest = json.loads((args.dataset_dir / 'manifest.json').read_text())
    task = dataset_manifest['task']
    initial_pose = dataset_manifest['initial_pose']
    training = load_sequences(args.dataset_dir, 'train')
    validation = load_sequences(args.dataset_dir, 'validation')
    if not sum(max(0, len(item[1]) - args.warmup) for item in training):
        raise ValueError('No scored training samples remain after warmup')
    if not sum(max(0, len(item[1]) - args.warmup) for item in validation):
        raise ValueError('No scored validation samples remain after warmup')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    network = BroadPredictor(hidden_size=args.hidden_size)
    initial_hash = None
    normalization_origin = 'train'
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
        if validate_checkpoint(initial) != args.hidden_size:
            raise ValueError('Initial checkpoint hidden size mismatch')
        artifact_sources(initial)
        if initial['config'].get('initial_pose') != initial_pose:
            raise ValueError('Initial checkpoint initial-pose domain mismatch')
        if initial['config'].get('task') != task:
            raise ValueError('Initial checkpoint motion task/horizon mismatch')
        network.load_state_dict(initial['state_dict'])
        normalization = initial['normalization']
        initial_hash = sha256(args.initial_checkpoint)
        normalization_origin = 'initial_checkpoint'
    else:
        sensors = np.concatenate([item[1] for item in training]).astype(np.float64)
        normalization = {'mean': sensors.mean(axis=0).tolist(),
                         'std': np.maximum(sensors.std(axis=0), 1e-8).tolist()}
    mean, std = (np.asarray(normalization[name], dtype=np.float64) for name in ('mean', 'std'))
    training = [(entry, torch.tensor((values - mean) / std, dtype=torch.float32),
                 torch.tensor(targets, dtype=torch.float32)) for entry, values, targets in training]
    validation = [(entry, torch.tensor((values - mean) / std, dtype=torch.float32),
                   torch.tensor(targets, dtype=torch.float64)) for entry, values, targets in validation]
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    optimizer_steps, scored = 0, 0
    history = []
    best_score = float('inf')
    selected_epoch, selected_steps, selected_scored = None, 0, 0
    selected_state = None
    for epoch in range(0 if args.initial_checkpoint else 1, args.epochs + 1):
        order, loss_sum, epoch_scored = [], 0., 0
        if epoch:
            shuffled = [training[index] for index in rng.permutation(len(training))]
            ordered = sorted(shuffled, key=lambda item: len(item[1]))
            order = [item[0]['sequence_id'] for item in ordered]
            network.train()
            for batch in batches(ordered, args.batch_size):
                values = torch.stack([item[1] for item in batch])
                targets = torch.stack([item[2] for item in batch])
                hidden = None
                for start in range(0, values.shape[1], args.chunk_size):
                    stop = min(start + args.chunk_size, values.shape[1])
                    mask = (torch.arange(start, stop) >= args.warmup)[None].expand(len(batch), -1)
                    if bool(mask.any()):
                        prediction, hidden = raw_sequence(network, values[:, start:stop], hidden)
                        loss = forecast_loss(prediction, targets[:, start:stop], mask)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite training loss')
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
                        optimizer.step()
                        count = int(mask.sum())
                        optimizer_steps += 1
                        scored += count
                        epoch_scored += count
                        loss_sum += float(loss.detach()) * count
                    else:
                        with torch.no_grad():
                            _, hidden = raw_sequence(network, values[:, start:stop], hidden)
                    hidden = hidden.detach()
        result = validation_metrics(network, validation, args.batch_size, args.warmup)
        history.append({'epoch': epoch, 'trajectory_order': order,
                        'training_loss': loss_sum / epoch_scored if epoch_scored else 0.,
                        'validation': result})
        if result['full_rmse_deg'] < best_score:
            best_score = result['full_rmse_deg']
            selected_epoch, selected_steps, selected_scored = epoch, optimizer_steps, scored
            selected_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
        print(f'Epoch {epoch}/{args.epochs}: loss={history[-1]["training_loss"]:.6f} '
              f'validation_rmse_deg={result["full_rmse_deg"]:.6f} selected={selected_epoch}',
              file=sys.stderr, flush=True)
    recipe = {name: sha256(ROOT / name) for name in (
        'model/motion_generate.py', 'model/motion_train.py', 'model/motion_evaluate.py',
        'model/orientation_generate.py', 'model/orientation_evaluate.py', 'host/orientation_sim.py')}
    config = {'input_size': 9, 'hidden_size': args.hidden_size, 'output_size': 4,
              'mode': 'default', 'backbone_layers': 0, 'dt': 1., 'warmup': args.warmup,
              'epochs': args.epochs, 'chunk_size': args.chunk_size, 'batch_size': args.batch_size,
              'seed': args.seed, 'learning_rate': args.learning_rate,
              'method': 'chronological_detached_tbptt', 'task': task, 'initial_pose': initial_pose,
              'loss_definition': '1-abs(dot(unit_raw,unit_target)) + raw_norm_penalty*(norm(raw)-1)^2',
              'raw_norm_penalty': .01}
    checkpoint = {'state_dict': selected_state, 'config': config,
                  'normalization': normalization, 'provenance': current_provenance(),
                  'training_recipe_sources': recipe, 'dataset_manifest_sha256': manifest_hash,
                  'selected_epoch': selected_epoch}
    if sha256(args.dataset_dir / 'manifest.json') != manifest_hash:
        raise ValueError('Dataset manifest changed during training')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / 'checkpoint.pt')
    report = {key: value for key, value in checkpoint.items() if key != 'state_dict'}
    report.update(optimizer_steps=optimizer_steps, training_samples_scored=scored,
                  selected_optimizer_steps=selected_steps, selected_training_samples_scored=selected_scored,
                  selected_validation_full_rmse_deg=best_score, history=history,
                  normalization_origin=normalization_origin, initial_checkpoint_sha256=initial_hash)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=positive_integer, default=10)
    parser.add_argument('--hidden-size', type=positive_integer, default=96)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--chunk-size', type=positive_integer, default=256)
    parser.add_argument('--batch-size', type=positive_integer, default=16)
    parser.add_argument('--initial-checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--learning-rate', type=positive_rate, default=.001)
    args = parser.parse_args()
    if args.seed < 0 or args.warmup < 0:
        parser.error('Seed and warmup must be nonnegative')
    try:
        train(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Motion training failed: {error}\n')


if __name__ == '__main__':
    main()
