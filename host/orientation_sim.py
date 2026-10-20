"""Fixed-time body-to-Earth pose steps and body-frame physical sensor values."""

import math

import numpy as np


MODEL_DT_S = .0035
MAGNETIC_DIRECTION = (-.02069254372432757, .35690462341735035, -.9339116170267592)


def normalize_quaternion(value):
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('Expected four finite quaternion components')
    norm = float(np.linalg.norm(quaternion))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError('Quaternion must have a finite nonzero norm')
    return quaternion / norm


def advance_pose(current, target, dt=MODEL_DT_S, max_rate=6):
    current, target = normalize_quaternion(current), normalize_quaternion(target)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_rate) or max_rate < 0:
        raise ValueError('Expected positive finite dt and nonnegative finite angular rate')
    dot = float(np.dot(current, target))
    if dot < 0:
        target = -target
        dot = -dot
    half_angle = math.acos(min(1., max(-1., dot)))
    if half_angle < 1e-12:
        return target.tolist()
    fraction = min(1., max_rate * dt / (2 * half_angle))
    result = (math.sin((1 - fraction) * half_angle) * current
              + math.sin(fraction * half_angle) * target) / math.sin(half_angle)
    return normalize_quaternion(result).tolist()


def sensors_for_step(previous, current, dt=MODEL_DT_S):
    previous, current = normalize_quaternion(previous), normalize_quaternion(current)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Expected a positive finite model timestep')
    w, x, y, z = current
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    a, b, c, d = previous * np.array([1, -1, -1, -1])
    delta = normalize_quaternion([a*w-b*x-c*y-d*z, a*x+b*w+c*z-d*y,
                                  a*y-b*z+c*w+d*x, a*z+b*y-c*x+d*w])
    if delta[0] < 0:
        delta = -delta
    magnitude = float(np.linalg.norm(delta[1:]))
    gyro = (np.zeros(3) if magnitude < 1e-14 else
            delta[1:] / magnitude * (2 * math.atan2(magnitude, delta[0]) / dt))
    magnetic = 45 * np.array(MAGNETIC_DIRECTION)
    return np.concatenate((rotation.T @ np.array([0., 0., 9.81]), gyro, rotation.T @ magnetic)).tolist()


def identity_sensors():
    return [0., 0., 9.81, 0., 0., 0., *[45 * value for value in MAGNETIC_DIRECTION]]
