"""Chronological detached-state TBPTT from an explicitly bound BROAD checkpoint."""

import argparse
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from ncps.torch import CfCCell

from model.broad_network import BroadPredictor
from model.broad_train import load_trials, positive_integer, quaternion_loss, sha256, validate


ROOT = Path(__file__).resolve().parents[1]


def current_provenance():
    sources = {f'model/{name}': sha256(ROOT / 'model' / name)
               for name in ('broad_train.py', 'broad_network.py')}
    sources['ncps/torch/cfc_cell.py'] = sha256(Path(inspect.getfile(CfCCell)))
    return {'sources': sources,
            'versions': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                         'numpy': np.__version__, 'ncps': importlib.metadata.version('ncps')}}


def load_initial(args):
    digest = sha256(args.initial_checkpoint)
    if digest != args.initial_checkpoint_sha256:
        raise ValueError('Initial checkpoint SHA256 identity mismatch')
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['provenance'] != current_provenance():
        raise ValueError('Initial checkpoint source/runtime provenance identity mismatch')
    if checkpoint['prepared_manifest_sha256'] != sha256(args.prepared_dir / 'manifest.json'):
        raise ValueError('Prepared training manifest SHA256 identity mismatch')
    config = checkpoint['config']
    for name, expected in {'input_size': 9, 'output_size': 4, 'mode': 'default',
                           'backbone_layers': 0, 'dt': 1.0}.items():
        if config.get(name) != expected:
            raise ValueError(f'Unsupported initial checkpoint configuration: {name}')
    if type(config.get('hidden_size')) is not int or config['hidden_size'] <= 0:
        raise ValueError('Initial hidden_size must be a positive integer')
    if type(config.get('warmup')) is not int or config['warmup'] < 0:
        raise ValueError('Initial warmup must be a nonnegative integer')
    mean, std = (np.asarray(checkpoint['normalization'][name], dtype=np.float64) for name in ('mean', 'std'))
    if mean.shape != (9,) or std.shape != (9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Initial normalization must have nine finite means and positive standard deviations')
    if any(not bool(torch.isfinite(value).all()) for value in checkpoint['state_dict'].values()):
        raise ValueError('Initial checkpoint weights must be finite')
    return checkpoint


def train(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    checkpoint = load_initial(args)
    recipe = {'model/broad_train_stateful.py': sha256(Path(__file__))}
    trials = load_trials(args.prepared_dir)
    training = [trial for trial in trials if trial['split'] == 'train']
    normalization = checkpoint['normalization']
    mean, std = (np.asarray(normalization[name], dtype=np.float64) for name in ('mean', 'std'))
    warmup = checkpoint['config']['warmup']
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    network = BroadPredictor(hidden_size=checkpoint['config']['hidden_size'])
    network.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    config = {'input_size': 9, 'hidden_size': checkpoint['config']['hidden_size'], 'output_size': 4,
              'mode': 'default', 'backbone_layers': 0, 'dt': 1.0, 'warmup': warmup,
              'loss': '1-squared_normalized_quaternion_dot', 'method': 'chronological_detached_tbptt',
              'seed': args.seed, 'epochs': args.epochs, 'chunk_size': args.chunk_size,
              'learning_rate': args.learning_rate, 'optimizer': 'Adam', 'gradient_clip': 1.0,
              'validation_every_pass': True, 'selection_includes_initial_checkpoint': True}
    entries, _ = validate(network, trials, normalization, warmup)
    initial_score = float(np.mean([entry['rmse_deg'] for entry in entries]))
    best_score, best_epoch, best_steps, best_loss = initial_score, 0, 0, None
    best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
    optimizer_steps, history = 0, []
    print(f'epoch 0: validation mean-trial RMSE {initial_score:.5f} deg', file=sys.stderr, flush=True)
    for epoch in range(1, args.epochs + 1):
        network.train()
        losses = []
        order = rng.permutation(len(training))
        for trial_index in order:
            trial = training[int(trial_index)]
            hidden = None
            for start in range(0, len(trial['sensors']), args.chunk_size):
                stop = min(start + args.chunk_size, len(trial['sensors']))
                inputs = torch.tensor(((trial['sensors'][start:stop] - mean) / std)[None], dtype=torch.float32)
                mask = (trial['movement'][start:stop] & trial['reference_valid'][start:stop]).copy()
                mask[np.arange(start, stop) < warmup] = False
                if mask.any():
                    predicted, hidden = network(inputs, hidden)
                    target = torch.tensor(trial['quaternion'][None, start:stop], dtype=torch.float32)
                    loss = quaternion_loss(predicted, target, torch.tensor(mask[None], dtype=torch.bool))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                    optimizer.step()
                    optimizer_steps += 1
                    losses.append(float(loss.detach()))
                else:
                    with torch.no_grad():
                        _, hidden = network(inputs, hidden)
                hidden = hidden.detach()
        entries, _ = validate(network, trials, normalization, warmup)
        score = float(np.mean([entry['rmse_deg'] for entry in entries]))
        training_loss = float(np.mean(losses)) if losses else None
        history.append({'epoch': epoch, 'validation_rmse_deg': score, 'training_loss': training_loss,
                        'optimizer_steps': optimizer_steps,
                        'trial_order': [training[int(index)]['trial_id'] for index in order]})
        print(f'epoch {epoch}/{args.epochs}: validation mean-trial RMSE {score:.5f} deg; '
              f'optimizer steps {optimizer_steps}', file=sys.stderr, flush=True)
        if score < best_score:
            best_score, best_epoch, best_steps, best_loss = score, epoch, optimizer_steps, training_loss
            best_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
    network.load_state_dict(best_state)
    entries, predictions = validate(network, trials, normalization, warmup)
    if sha256(args.initial_checkpoint) != args.initial_checkpoint_sha256 or current_provenance() != checkpoint['provenance']:
        raise ValueError('Initial checkpoint or runtime provenance changed during training')
    if sha256(args.prepared_dir / 'manifest.json') != checkpoint['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest identity changed during training')
    if recipe['model/broad_train_stateful.py'] != sha256(Path(__file__)):
        raise ValueError('Stateful training recipe source changed during training')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    saved = {'state_dict': best_state, 'config': config, 'normalization': normalization,
             'prepared_manifest_sha256': checkpoint['prepared_manifest_sha256'],
             'provenance': checkpoint['provenance'], 'training_recipe_sources': recipe,
             'initial_checkpoint_sha256': args.initial_checkpoint_sha256,
             'best_epoch': best_epoch, 'optimizer_steps': best_steps}
    torch.save(saved, output / 'checkpoint.pt')
    for entry, (trial, predicted) in zip(entries, predictions):
        path = output / entry['predictions_file']
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, sensors=trial['sensors'], targets=trial['quaternion'], predictions=predicted,
                            movement=trial['movement'], reference_valid=trial['reference_valid'],
                            sample_index=trial['sample_index'])
    report = {key: saved[key] for key in ('config', 'normalization', 'prepared_manifest_sha256', 'provenance',
              'training_recipe_sources', 'initial_checkpoint_sha256', 'best_epoch')}
    report.update(parameter_count=sum(p.numel() for p in network.parameters()), optimizer_steps=optimizer_steps,
                  best_optimizer_steps=best_steps, training_loss=best_loss, history=history,
                  initial_validation_rmse_deg=initial_score,
                  validation_rmse_deg=float(np.mean([entry['rmse_deg'] for entry in entries])), validation=entries)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output_dir': str(output), 'best_epoch': best_epoch,
                      'validation_rmse_deg': report['validation_rmse_deg']}))


def positive_rate(value):
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be finite and positive')
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-dir', 'initial-checkpoint', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--initial-checkpoint-sha256', required=True)
    parser.add_argument('--epochs', type=positive_integer, default=10)
    parser.add_argument('--chunk-size', type=positive_integer, default=256)
    parser.add_argument('--learning-rate', type=positive_rate, default=0.0003)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    try:
        train(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Stateful BROAD training failed: {error}\n')


if __name__ == '__main__':
    main()
