"""Vectorized integer-only BROAD recurrence: Q11 inputs and Q13 model words."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NUMERIC = {'word_bits': 16, 'input_fractional_bits': 11, 'state_fractional_bits': 13,
           'weight_fractional_bits': 13, 'output_fractional_bits': 13, 'accumulator_bits': 48,
           'accumulator_fractional_bits': 26, 'rounding': 'half_away_from_zero',
           'activation_entries': 4096, 'activation_fractional_bits': 8, 'activation_zero_index': 2048}
FILES = ('weights.hex', 'tanh.hex', 'sigmoid.hex', 'float_checkpoint.pt')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_versions():
    import torch
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__),
            'numpy': np.__version__, 'ncps': importlib.metadata.version('ncps')}


def float_sources():
    from ncps.torch import CfCCell
    result = {f'model/{name}': sha256(ROOT / 'model' / name) for name in ('broad_train.py', 'broad_network.py')}
    result['ncps/torch/cfc_cell.py'] = sha256(Path(inspect.getfile(CfCCell)))
    return result


def artifact_sources(checkpoint):
    result = float_sources()
    for name in ('broad_integer.py', 'broad_export_integer.py', 'broad_evaluate_integer.py', 'broad_evaluate.py'):
        result[f'model/{name}'] = sha256(ROOT / 'model' / name)
    for name, expected in checkpoint.get('training_recipe_sources', {}).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(ROOT / path) != expected:
            raise ValueError('Training recipe source SHA256 identity mismatch')
        result[name] = expected
    return result


def validate_checkpoint(checkpoint):
    import torch
    config = checkpoint['config']
    for key, value in {'input_size': 9, 'output_size': 4, 'mode': 'default',
                       'backbone_layers': 0, 'dt': 1.0}.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported checkpoint configuration: {key}')
    size = config['hidden_size']
    if type(size) is not int or size <= 0 or type(config['warmup']) is not int or config['warmup'] < 0:
        raise ValueError('Invalid hidden size or warmup')
    mean, std = (np.asarray(checkpoint['normalization'][key], dtype=np.float64) for key in ('mean', 'std'))
    if mean.shape != (9,) or std.shape != (9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid checkpoint normalization')
    if checkpoint['provenance']['sources'] != float_sources() or checkpoint['provenance']['versions'] != runtime_versions():
        raise ValueError('Checkpoint source/runtime identity mismatch')
    if any(not bool(torch.isfinite(value).all()) for value in checkpoint['state_dict'].values()):
        raise ValueError('Nonfinite checkpoint coefficients')
    return size


def round_shift(values, bits):
    magnitude = (np.abs(values) + (1 << (bits - 1))) >> bits
    return np.where(values < 0, -magnitude, magnitude)


def quantize_inputs(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Normalized inputs must be finite')
    with np.errstate(over='ignore'):
        rounded = np.copysign(np.floor(np.abs(values) * 2048 + 0.5), values)
    clipped = np.clip(rounded, -32768, 32767)
    return clipped.astype(np.int16), int(np.count_nonzero(rounded != clipped))


def read_words(path):
    words = path.read_text().split()
    if any(len(word) != 4 or any(c not in '0123456789abcdefABCDEF' for c in word) for word in words):
        raise ValueError(f'Invalid signed-16 hex file: {path.name}')
    values = np.array([int(word, 16) for word in words], dtype=np.int64)
    return np.where(values >= 32768, values - 65536, values)


def check_accumulator(values):
    if np.any(values < -(1 << 47)) or np.any(values >= (1 << 47)):
        raise ValueError('Signed 48-bit accumulator overflow')


class GoldenBroadPredictor:
    def __init__(self, hidden_size, weights, tanh, sigmoid):
        self.hidden_size = hidden_size
        arrays = [np.asarray(values) for values in (weights, tanh, sigmoid)]
        expected = (4 * hidden_size * (hidden_size + 10) + 4 * (hidden_size + 1), 4096, 4096)
        for values, count in zip(arrays, expected):
            if values.shape != (count,) or values.dtype.kind not in 'iu' or np.any(values < -32768) or np.any(values > 32767):
                raise ValueError('Malformed signed-16 coefficient or activation arrays')
        weights, self.tanh, self.sigmoid = [values.astype(np.int64, copy=True) for values in arrays]
        if np.any(np.abs(self.tanh) > 8192) or np.any(self.sigmoid < 0) or np.any(self.sigmoid > 8192):
            raise ValueError('Activation values outside tanh/sigmoid Q13 ranges')
        base = 4 * hidden_size * (hidden_size + 10)
        heads = weights[:base].reshape(4 * hidden_size, hidden_size + 10)
        self.input_weights = heads[:, :9].copy()
        self.hidden_weights = heads[:, 9:-1].copy()
        self.biases = heads[:, -1].copy() << 13
        output = weights[base:].reshape(4, hidden_size + 1)
        self.output_weights = output[:, :-1].copy()
        self.output_biases = output[:, -1].copy() << 13
        self.reset()

    @classmethod
    def from_export(cls, directory):
        import torch
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['numeric'] != NUMERIC or set(manifest['files']) != set(FILES):
            raise ValueError('Unsupported numeric format or artifact file set')
        for filename in FILES:
            if sha256(directory / filename) != manifest['files'][filename]:
                raise ValueError(f'Artifact SHA256 identity mismatch: {filename}')
        if manifest['files']['float_checkpoint.pt'] != manifest['checkpoint_sha256']:
            raise ValueError('Copied checkpoint SHA256 identity mismatch')
        checkpoint = torch.load(directory / 'float_checkpoint.pt', map_location='cpu', weights_only=True)
        size = validate_checkpoint(checkpoint)
        if manifest['config'] != checkpoint['config'] or manifest['normalization'] != checkpoint['normalization']:
            raise ValueError('Artifact/checkpoint configuration mismatch')
        if manifest['sources'] != artifact_sources(checkpoint) or manifest['versions'] != runtime_versions():
            raise ValueError('Artifact source/runtime identity mismatch')
        predictor = cls(size, *(read_words(directory / name) for name in FILES[:3]))
        predictor.manifest = manifest
        predictor.checkpoint = checkpoint
        return predictor

    def reset(self):
        self.hidden = np.zeros(self.hidden_size, dtype=np.int64)
        self.counters = dict(lut_clamp_count=0, state_clipping_count=0, raw_clipping_count=0)

    def lookup(self, table, values):
        index = round_shift(values, 18) + 2048
        self.counters['lut_clamp_count'] += int(np.count_nonzero((index < 0) | (index > 4095)))
        return table[np.clip(index, 0, 4095)]

    def step(self, inputs):
        inputs = np.asarray(inputs)
        if inputs.shape != (9,) or inputs.dtype.kind not in 'iu' or np.any(inputs < -32768) or np.any(inputs > 32767):
            raise ValueError('Expected nine signed-16 integer inputs')
        inputs = inputs.astype(np.int64)
        heads = self.biases + ((self.input_weights @ inputs) << 2) + self.hidden_weights @ self.hidden
        check_accumulator(heads)
        heads = heads.reshape(4, self.hidden_size)
        gate_acc = heads[2] + heads[3]
        check_accumulator(gate_acc)
        f1, f2 = self.lookup(self.tanh, heads[0]), self.lookup(self.tanh, heads[1])
        gate = self.lookup(self.sigmoid, gate_acc)
        unbounded = f1 + round_shift((f2 - f1) * gate, 13)
        hidden = np.clip(unbounded, -32768, 32767)
        self.counters['state_clipping_count'] += int(np.count_nonzero(hidden != unbounded))
        accumulators = self.output_biases + self.output_weights @ hidden
        check_accumulator(accumulators)
        unbounded = round_shift(accumulators, 13)
        raw = np.clip(unbounded, -32768, 32767)
        self.counters['raw_clipping_count'] += int(np.count_nonzero(raw != unbounded))
        self.hidden = hidden
        return hidden.tolist(), raw.tolist()
