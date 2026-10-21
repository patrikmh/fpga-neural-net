"""Local future-motion viewer using the same verified H96 USB protocol."""

import json
import math

from host.orientation_demo import OrientationSession, main as serve
from host.orientation_sim import MODEL_DT_S
from model.motion_generate import task_identity


class MotionSession(OrientationSession):
    def __init__(self, args):
        # Reject an orientation checkpoint before any device can be opened.
        manifest = json.loads((args.export_dir / 'manifest.json').read_text())
        self.task = manifest.get('config', {}).get('task', {})
        horizon = self.task.get('horizon_s')
        if (type(horizon) not in (int, float) or not math.isfinite(horizon) or horizon <= 0
                or self.task != task_identity(horizon)):
            raise ValueError('Motion demo requires an explicitly trained motion_prediction export')
        super().__init__(args)

    def snapshot(self, status, count, reset_verified, sample=None):
        return {**super().snapshot(status, count, reset_verified, sample), 'task': self.task.copy()}

    def step(self, target_wxyz=None):
        state = super().step(target_wxyz)
        sample = state['sample']
        sample.pop('error_deg')
        sample['forecast'] = {'status': 'pending', 'horizon_s': self.task['horizon_s'],
                              'issued_at_s': sample['model_time_s'],
                              'target_time_s': sample['model_time_s'] + self.task['horizon_s'],
                              'truth_wxyz': None, 'error_deg': None}
        return state


def main():
    serve(session_factory=MotionSession)


if __name__ == '__main__':
    main()
