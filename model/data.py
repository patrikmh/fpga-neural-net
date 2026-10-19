"""Independent deterministic signal trajectories and training-only normalization."""

import numpy as np


GENERATOR_CONFIG = {
    "families": ["sinusoidal_mixture", "linear_frequency_sweep"],
    "frequency_cycles_per_sample": [0.02, 0.12],
    "amplitude": [0.3, 1.0],
    "noise_std": 0.005,
}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


def signal_trajectories(seed, split, count, length):
    split_id = SPLIT_IDS[split]
    trajectories = []
    t = np.arange(length, dtype=np.float64)
    for trajectory_id in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, trajectory_id]))
        amplitude = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        if trajectory_id % 2 == 0:
            frequencies = rng.uniform(0.02, 0.12, size=2)
            second_amplitude = rng.uniform(0.3, 1.0)
            second_phase = rng.uniform(0, 2 * np.pi)
            signal = amplitude * np.sin(2 * np.pi * frequencies[0] * t + phase)
            signal += second_amplitude * np.sin(2 * np.pi * frequencies[1] * t + second_phase)
        else:
            start, end = rng.uniform(0.02, 0.12, size=2)
            rate = (end - start) / max(length - 1, 1)
            signal = amplitude * np.sin(2 * np.pi * (start * t + 0.5 * rate * t * t) + phase)
        signal += rng.normal(0, 0.005, size=length)
        trajectories.append(signal)
    return np.asarray(trajectories, dtype=np.float32)


def fitnormalization(trajectories):
    mean = float(np.mean(trajectories, dtype=np.float64))
    std = float(np.std(trajectories, dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Training trajectories must have finite positive standard deviation")
    return {"mean": mean, "std": std}
