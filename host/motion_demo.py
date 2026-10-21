"""Local future-motion viewer using the same verified H96 USB protocol."""

from collections import deque
from copy import deepcopy
import json
import math

import numpy as np

from host.orientation_demo import OrientationSession, main as serve
from host.orientation_sim import MODEL_DT_S, normalize_quaternion
from model.motion_generate import task_identity
from model.motion_evaluate import sensor_baselines


class MotionSession(OrientationSession):
    def __init__(self, args):
        # Reject an orientation checkpoint before any device can be opened.
        manifest = json.loads((args.export_dir / 'manifest.json').read_text())
        self.task = manifest.get('config', {}).get('task', {})
        horizon = self.task.get('horizon_s')
        if (type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0
                or self.task != task_identity(horizon)):
            raise ValueError('Motion demo requires an explicitly trained motion_prediction export')
        self.pending_forecasts = deque()
        super().__init__(args)

    def snapshot(self, status, count, reset_verified, sample=None):
        return {**super().snapshot(status, count, reset_verified, sample), 'task': self.task.copy(),
                'pending_forecast_count': len(self.pending_forecasts)}

    def reset(self):
        state = super().reset()
        self.pending_forecasts.clear()
        state['pending_forecast_count'] = 0
        return state

    def disconnect(self):
        state = super().disconnect()
        self.pending_forecasts.clear()
        state['pending_forecast_count'] = 0
        return state

    def step(self, target_wxyz=None):
        previous_truth = self.truth.copy()
        state = super().step(target_wxyz)
        sample = state['sample']
        sample.pop('error_deg')
        normalization = self.predictor.manifest['normalization']
        mean, std = (np.asarray(normalization[name], dtype=np.float64) for name in ('mean', 'std'))
        effective_sensors = np.asarray(sample['quantized_inputs'], dtype=np.float64) / 2048 * std + mean
        baselines = {name: {'quaternion_wxyz': values[0].tolist(), 'error_deg': None}
                     for name, values in sensor_baselines(effective_sensors[None], self.task['horizon_s']).items()}
        forecast = {'status': 'pending', 'issued_index': sample['index'],
                    'horizon_s': self.task['horizon_s'], 'issued_at_s': sample['model_time_s'],
                    'target_time_s': sample['model_time_s'] + self.task['horizon_s'],
                    'quaternion_wxyz': sample['quaternion_wxyz'].copy(),
                    'raw_outputs': sample['raw_outputs'].copy(), 'baselines': baselines,
                    'truth_wxyz': None, 'error_deg': None}
        self.pending_forecasts.append(deepcopy(forecast))
        sample['forecast'] = forecast
        sample['scored_forecast'] = None
        while (self.pending_forecasts
               and self.pending_forecasts[0]['target_time_s'] <= sample['model_time_s'] + 1e-12):
            matured = self.pending_forecasts.popleft()
            previous_time = sample['model_time_s'] - MODEL_DT_S
            fraction = (matured['target_time_s'] - previous_time) / MODEL_DT_S
            truth = interpolate_truth(previous_truth, sample['truth_wxyz'], min(1., max(0., fraction)))
            matured.update(status='scored', truth_wxyz=truth,
                           error_deg=angle_error(matured['quaternion_wxyz'], truth))
            for baseline in matured['baselines'].values():
                baseline['error_deg'] = angle_error(baseline['quaternion_wxyz'], truth)
            sample['scored_forecast'] = matured
        state['pending_forecast_count'] = len(self.pending_forecasts)
        return state


def angle_error(prediction, truth):
    dot = abs(float(np.dot(normalize_quaternion(prediction), normalize_quaternion(truth))))
    return math.degrees(2 * math.acos(min(1., max(0., dot))))


def interpolate_truth(previous, current, fraction):
    previous, current = normalize_quaternion(previous), normalize_quaternion(current)
    dot = float(np.dot(previous, current))
    if dot < 0:
        current, dot = -current, -dot
    half_angle = math.acos(min(1., max(-1., dot)))
    if half_angle < 1e-10:
        result = (1 - fraction) * previous + fraction * current
    else:
        result = (math.sin((1 - fraction) * half_angle) * previous
                  + math.sin(fraction * half_angle) * current) / math.sin(half_angle)
    return normalize_quaternion(result).tolist()


def main():
    serve(session_factory=MotionSession)


if __name__ == '__main__':
    main()
