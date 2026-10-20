"""Exploratory six-axis tilt baseline; no training or FPGA changes."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'build/datasets/broad-preview'
TAU_S = 0.5  # Fixed before reading any optical reference results.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'build/broad-preview/result.json')
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True, exist_ok=True)


def propagate(up, omega, dt):
    rate = -omega
    speed = float(np.linalg.norm(rate))
    if speed < 1e-12:
        return up.copy()
    axis = rate / speed
    angle = speed * dt
    return (up * np.cos(angle) + np.cross(axis, up) * np.sin(angle)
            + axis * np.dot(axis, up) * (1 - np.cos(angle)))


def angular_rmse(predicted, reference, mask):
    dot = np.sum(predicted[mask] * reference[mask], axis=1)
    errors = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return float(np.sqrt(np.mean(errors ** 2)))


assert np.allclose(propagate(np.array([0., 0., 1.]), np.array([0., 0., 1.]), .1), [0, 0, 1])
assert np.allclose(propagate(np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.pi / 2), [0, 1, 0])
results = []
for path in sorted((DATA / 'data_hdf5').glob('*.hdf5')):
    with h5py.File(path, 'r') as source:
        acc, gyr, mag = (source[name][:] for name in ('imu_acc', 'imu_gyr', 'imu_mag'))
        q, movement = source['opt_quat'][:], source['movement'][:].astype(bool)
        rate_hz = float(source.attrs['sampling_rate'])
    assert all(array.shape == acc.shape for array in (gyr, mag))
    assert all(np.isfinite(array).all() for array in (acc, gyr, mag)), 'Invalid sensor samples: do not silently collapse time'
    acc_norm = np.linalg.norm(acc, axis=1)
    assert np.all(acc_norm > 0)
    measured_up = acc / acc_norm[:, None]
    norm = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norm > 1e-12)
    q = q / np.where(valid, norm, 1)[:, None]
    w, x, y, z = q.T
    reference_up = np.column_stack((2 * (x*z-w*y), 2 * (y*z+w*x), 1-2*(x*x+y*y)))
    mask = movement & valid
    assert mask.any()
    dt = 1 / rate_hz
    alpha = 1 - np.exp(-dt / TAU_S)
    fused, gyro = np.empty_like(acc), np.empty_like(acc)
    fused[0] = gyro[0] = measured_up[0]  # Never initialize from optical ground truth.
    for index in range(1, len(acc)):
        omega = (gyr[index-1] + gyr[index]) * .5
        gyro[index] = propagate(gyro[index-1], omega, dt)
        gyro[index] /= np.linalg.norm(gyro[index])
        prior = propagate(fused[index-1], omega, dt)
        fused[index] = (1-alpha) * prior + alpha * measured_up[index]
        fused[index] /= np.linalg.norm(fused[index])
    result = {
        'trial': path.stem, 'samples': len(acc), 'sampling_rate_hz': rate_hz,
        'movement_samples_scored': int(mask.sum()), 'invalid_optical_samples_excluded': int((movement & ~valid).sum()),
        'sensor_values_finite': True, 'quaternion_norm_max_deviation': float(np.max(np.abs(norm[valid]-1))),
        'inclination_rmse_deg': {
            'accelerometer_only': angular_rmse(measured_up, reference_up, mask),
            'gyro_only_acc_initialization': angular_rmse(gyro, reference_up, mask),
            'gyro_acc_complementary': angular_rmse(fused, reference_up, mask),
        },
    }
    results.append(result)
    print(json.dumps(result), flush=True)
report = {
    'scope': 'Exploratory CPU six-axis tilt baseline; no CfC training, heading estimate, or FPGA sensor fusion',
    'source': 'https://github.com/dlaidig/broad', 'dataset_commit': '7e2f8189bb4a3ac91eae32c69b6bc87f08a6e2ad',
    'license': 'CC-BY-4.0', 'complementary_time_constant_s': TAU_S,
    'reference_convention': 'opt_quat wxyz body-to-ENU; compare Earth-up expressed in sensor coordinates',
    'scoring': 'angular inclination RMSE on movement and finite nonzero reference quaternions; retain all time steps',
    'model_selection': 'No tuning; these preview trials are not held-out tests',
    'versions': {name: importlib.metadata.version(name) for name in ('h5py', 'numpy')},
    'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'download_manifest_sha256': hashlib.sha256((DATA / 'download-manifest.json').read_bytes()).hexdigest(),
    'trials': results,
}
with args.output.open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
