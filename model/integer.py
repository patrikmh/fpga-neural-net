"""Integer-only Q13 reference for the exported eight-neuron CfC."""

import hashlib
import json
import math
from pathlib import Path


NUMERIC = {
    "word_bits": 16, "fractional_bits": 13, "accumulator_bits": 48,
    "accumulator_fractional_bits": 26, "rounding": "half_away_from_zero",
    "activation_entries": 4096, "activation_fractional_bits": 8,
    "activation_zero_index": 2048,
}


def half_away(value):
    if not math.isfinite(value):
        raise ValueError("Cannot quantize a nonfinite value")
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def saturate(value):
    return min(32767, max(-32768, value))


def quantize(value):
    return saturate(half_away(value * 8192))


def round_shift(value, shift):
    if shift == 0:
        return value
    if shift < 0:
        raise ValueError("Shift must be nonnegative")
    magnitude = (abs(value) + (1 << (shift - 1))) >> shift
    return magnitude if value >= 0 else -magnitude


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_words(path):
    words = path.read_text().split()
    if any(len(word) != 4 or any(c not in "0123456789abcdefABCDEF" for c in word) for word in words):
        raise ValueError(f"Invalid hexadecimal words in {path}")
    return [int(word, 16) - (65536 if int(word, 16) >= 32768 else 0) for word in words]


class GoldenIntegerPredictor:
    def __init__(self, weights, tanh, sigmoid):
        if (len(weights), len(tanh), len(sigmoid)) != (329, 4096, 4096):
            raise ValueError("Invalid coefficient or activation table lengths")
        if any(type(value) is not int or not -32768 <= value <= 32767
               for values in (weights, tanh, sigmoid) for value in values):
            raise ValueError("Coefficients and activation tables must contain signed 16-bit integers")
        if any(not -8192 <= value <= 8192 for value in tanh):
            raise ValueError("Tanh activation values must be in the Q13 range [-8192,8192]")
        if any(not 0 <= value <= 8192 for value in sigmoid):
            raise ValueError("Sigmoid activation values must be in the Q13 range [0,8192]")
        self.weights = list(weights)
        self.tanh = list(tanh)
        self.sigmoid = list(sigmoid)
        self.reset()

    @classmethod
    def from_export(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["numeric"] != NUMERIC:
            raise ValueError("Unsupported integer numeric format")
        if manifest["sources"]["model/integer.py"] != sha256(Path(__file__)):
            raise ValueError("Integer source SHA256 identity mismatch")
        required = {"weights.hex", "tanh.hex", "sigmoid.hex", "float_checkpoint.pt"}
        if not required <= set(manifest["files"]):
            raise ValueError("Export manifest is missing required files")
        for filename, expected in manifest["files"].items():
            if Path(filename).name != filename or sha256(directory / filename) != expected:
                raise ValueError(f"Export SHA256 identity mismatch: {filename}")
        if manifest["files"]["float_checkpoint.pt"] != manifest["checkpoint_sha256"]:
            raise ValueError("Float checkpoint SHA256 identity mismatch")
        return cls(*(read_words(directory / name) for name in ("weights.hex", "tanh.hex", "sigmoid.hex")))

    def reset(self):
        self.hidden = [0] * 8
        self.output_clipping_count = 0

    def step(self, value):
        if type(value) is not int or not -32768 <= value <= 32767:
            raise ValueError("Input must be a signed 16-bit integer")
        vector = [value] + self.hidden
        heads = []
        for head in range(4):
            rows = []
            for neuron in range(8):
                offset = head * 80 + neuron * 10
                rows.append((self.weights[offset + 9] << 13)
                            + sum(vector[column] * self.weights[offset + column] for column in range(9)))
            heads.append(rows)
        new_hidden = []
        for neuron in range(8):
            f1 = self.tanh[min(4095, max(0, round_shift(heads[0][neuron], 18) + 2048))]
            f2 = self.tanh[min(4095, max(0, round_shift(heads[1][neuron], 18) + 2048))]
            gate_acc = heads[2][neuron] + heads[3][neuron]
            gate = self.sigmoid[min(4095, max(0, round_shift(gate_acc, 18) + 2048))]
            new_hidden.append(saturate(f1 + round_shift((f2 - f1) * gate, 13)))
        acc = (self.weights[328] << 13) + sum(new_hidden[i] * self.weights[320 + i] for i in range(8))
        rounded = round_shift(acc, 13)
        output = saturate(rounded)
        self.output_clipping_count += int(output != rounded)
        self.hidden = new_hidden
        return list(new_hidden), output
