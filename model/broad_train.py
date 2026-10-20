"""Bounded CPU orientation training on declared BROAD train/validation trials."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from ncps.torch import CfCCell

from model.broad_network import BroadPredictor


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_normalization(trials):
    sensors = np.concatenate([trial['sensors'] for trial in trials if trial['split'] == 'train']).astype(np.float64)
    if sensors.ndim != 2 or sensors.shape[1] != 9 or not np.isfinite(sensors).all():
        raise ValueError('Expected finite nine-channel training sensors')
    mean, std = sensors.mean(axis=0), sensors.std(axis=0)
    constant = std < 1e-8
    std[constant] = 1.0
    return {'mean': mean.tolist(), 'std': std.tolist(), 'constant_channels': np.flatnonzero(constant).tolist()}


def quaternion_loss(prediction, target, mask):
    if not bool(mask.any()):
        return prediction[mask].sum() * 0
    predicted = F.normalize(prediction[mask], dim=-1)
    reference = F.normalize(target[mask], dim=-1)
    dot = (predicted * reference).sum(dim=-1).clamp(-1, 1)
    return (1 - dot.square()).mean()


def orientation_errors_deg(prediction, target):
    predicted = F.normalize(prediction.to(torch.float64), dim=-1)
    reference = F.normalize(target.to(torch.float64), dim=-1)
    dot = (predicted * reference).sum(dim=-1).abs().clamp(0, 1)
    return torch.rad2deg(2 * torch.acos(dot)).to(prediction.dtype)


def load_trials(prepared):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    trials = []
    for entry in manifest['spec']['trials']:
        if entry['split'] not in ('train', 'validation'):
            continue
        trial_id = int(entry['id'])
        filename = f"{entry['split']}/{trial_id:02d}.npz"
        path = prepared / filename
        if sha256(path) != manifest['outputs'][filename]:
            raise ValueError(f'Prepared trial SHA256 mismatch: {filename}')
        with np.load(path, allow_pickle=False) as saved:
            trial = {name: saved[name].copy() for name in
                     ('sensors', 'quaternion', 'movement', 'reference_valid', 'sample_index')}
        count = len(trial['sensors'])
        if trial['sensors'].shape != (count, 9) or trial['quaternion'].shape != (count, 4):
            raise ValueError(f'Invalid trial dimensions: {filename}')
        if any(trial[name].shape != (count,) for name in ('movement', 'reference_valid', 'sample_index')):
            raise ValueError(f'Invalid trial masks/index: {filename}')
        if not np.isfinite(trial['sensors']).all():
            raise ValueError(f'Nonfinite sensors: {filename}')
        trial.update(split=entry['split'], trial_id=trial_id)
        trials.append(trial)
    if not any(t['split'] == 'train' for t in trials) or not any(t['split'] == 'validation' for t in trials):
        raise ValueError('Training and validation trials are both required')
    return trials


def predict_trial(network, sensors, normalization, chunk_size=1024):
    """Predict every timestep, carrying state across memory-bounded chunks."""
    mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
    normalized = (sensors - mean) / std
    hidden, chunks = None, []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), chunk_size):
            inputs = torch.tensor(normalized[None, start:start + chunk_size], dtype=torch.float32)
            predicted, hidden = network(inputs, hidden)
            chunks.append(predicted[0].numpy())
    return np.concatenate(chunks)


def validate(network, trials, normalization, warmup):
    entries, predictions = [], []
    with torch.no_grad():
        for trial in trials:
            if trial['split'] != 'validation':
                continue
            full = predict_trial(network, trial['sensors'], normalization)
            mask = (trial['movement'] & trial['reference_valid']).copy()
            mask[:warmup] = False
            if not mask.any():
                raise ValueError(f"No validation samples remain for trial {trial['trial_id']}")
            errors = orientation_errors_deg(torch.tensor(full[mask], dtype=torch.float64),
                                            torch.tensor(trial['quaternion'][mask], dtype=torch.float64))
            rmse = float(torch.sqrt(torch.mean(errors.square())))
            entries.append({'trial_id': trial['trial_id'], 'samples_scored': int(mask.sum()),
                            'rmse_deg': rmse, 'predictions_file': f"validation/{trial['trial_id']:02d}.npz"})
            predictions.append((trial, full))
    return entries, predictions


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    prepared_hash = sha256(args.prepared_dir / 'manifest.json')
    trials = load_trials(args.prepared_dir)
    normalization = fit_normalization(trials)
    training = [trial for trial in trials if trial['split'] == 'train']
    eligible = []
    for trial in training:
        scored = trial['movement'] & trial['reference_valid']
        count = len(scored) - args.window + 1
        if count <= 0:
            raise ValueError('Training trial is shorter than the requested window')
        prefix = np.concatenate(([0], np.cumsum(scored)))
        starts = np.arange(count)
        eligible.append(starts[prefix[starts + args.window] - prefix[starts + args.warmup] > 0])
    if any(len(starts) == 0 for starts in eligible):
        raise ValueError('Training trial has no scored windows')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    network = BroadPredictor()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    config = {'seed': args.seed, 'epochs': args.epochs, 'updates_per_epoch': args.updates_per_epoch,
              'batch_size': args.batch_size, 'window': args.window, 'warmup': args.warmup,
              'validation_every': args.validation_every, 'learning_rate': 0.003, 'gradient_clip': 1.0,
              'input_size': 9, 'hidden_size': 8, 'output_size': 4, 'mode': 'default',
              'backbone_layers': 0, 'dt': 1.0, 'loss': '1-squared_normalized_quaternion_dot'}
    mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
    best_score = float('inf')
    for epoch in range(1, args.epochs + 1):
        network.train()
        losses = []
        for _ in range(args.updates_per_epoch):
            inputs, targets, masks = [], [], []
            for _ in range(args.batch_size):
                index = int(rng.integers(len(training)))
                trial = training[index]
                start = int(rng.choice(eligible[index]))
                stop = start + args.window
                inputs.append((trial['sensors'][start:stop] - mean) / std)
                targets.append(trial['quaternion'][start:stop])
                mask = (trial['movement'][start:stop] & trial['reference_valid'][start:stop]).copy()
                mask[:args.warmup] = False
                masks.append(mask)
            optimizer.zero_grad()
            predicted, _ = network(torch.tensor(np.stack(inputs), dtype=torch.float32))
            loss = quaternion_loss(predicted, torch.tensor(np.stack(targets), dtype=torch.float32),
                                   torch.tensor(np.stack(masks), dtype=torch.bool))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs:
            entries, _ = validate(network, trials, normalization, args.warmup)
            score = float(np.mean([entry['rmse_deg'] for entry in entries]))
            print(f'epoch {epoch}/{args.epochs}: validation mean-trial RMSE {score:.5f} deg', file=sys.stderr, flush=True)
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
                best_loss = float(np.mean(losses))
    network.load_state_dict(best_state)
    entries, predictions = validate(network, trials, normalization, args.warmup)
    provenance = {
        'sources': {f'model/{name}': sha256(Path(__file__).with_name(name))
                    for name in ('broad_train.py', 'broad_network.py')},
        'versions': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                     'numpy': np.__version__, 'ncps': importlib.metadata.version('ncps')},
    }
    provenance['sources']['ncps/torch/cfc_cell.py'] = sha256(Path(inspect.getfile(CfCCell)))
    if sha256(args.prepared_dir / 'manifest.json') != prepared_hash:
        raise ValueError('Prepared manifest changed during training')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'config': config, 'normalization': normalization,
                'prepared_manifest_sha256': prepared_hash, 'provenance': provenance,
                'best_epoch': best_epoch}, output / 'checkpoint.pt')
    for entry, (trial, predicted) in zip(entries, predictions):
        path = output / entry['predictions_file']
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, sensors=trial['sensors'], targets=trial['quaternion'], predictions=predicted,
                            movement=trial['movement'], reference_valid=trial['reference_valid'],
                            sample_index=trial['sample_index'])
    report = {'config': config, 'parameter_count': sum(p.numel() for p in network.parameters()),
              'training_loss': best_loss, 'best_epoch': best_epoch, 'validation': entries,
              'validation_rmse_deg': float(np.mean([entry['rmse_deg'] for entry in entries])),
              'prepared_manifest_sha256': prepared_hash, 'provenance': provenance}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'validation_rmse_deg': report['validation_rmse_deg']}))


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Value must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    for name, default in (('epochs', 50), ('updates-per-epoch', 16), ('batch-size', 16),
                          ('window', 256), ('warmup', 32), ('validation-every', 10)):
        parser.add_argument('--' + name, type=positive_integer, default=default)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.seed < 0 or args.warmup >= args.window:
        parser.error('Seed must be nonnegative and warmup must be shorter than the window')
    try:
        train(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'BROAD training failed: {error}\n')


if __name__ == '__main__':
    main()
