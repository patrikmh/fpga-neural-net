"""Package an old validation motion window for numerical USB equivalence."""
import json
from pathlib import Path
import sys

import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root))
from model.broad_evaluate import current_identities
from model.broad_integer import GoldenBroadPredictor, quantize_inputs, sha256

selection_path = root / 'artifacts/broad-h96-final-v1/selection.json'
selection = json.loads(selection_path.read_text())
sources, versions = current_identities()
assert sources == selection['model_sources'] and versions == selection['versions']
export = root / 'artifacts/broad-integer-v1/export'
predictor = GoldenBroadPredictor.from_export(export)
assert sha256(export / 'manifest.json') == selection['integer_export_manifest_sha256']
prepared = root / 'build/broad-v1-prepared'
assert sha256(prepared / 'manifest.json') == predictor.checkpoint['prepared_manifest_sha256']
trial = prepared / 'validation/02.npz'
trial_hash = sha256(trial)
with np.load(trial, allow_pickle=False) as arrays:
    start = int(np.flatnonzero(arrays['movement'])[0])
    count = 1000
    sensors = arrays['sensors'][start:start + count].copy()
    movement = arrays['movement'][start:start + count].copy()
    valid = arrays['reference_valid'][start:start + count].copy()
assert len(sensors) == count and int(movement.sum()) == count
mean, std = (np.asarray(predictor.manifest['normalization'][key], dtype=np.float64)
             for key in ('mean', 'std'))
inputs, clips = quantize_inputs((sensors.astype(np.float64) - mean) / std)
assert clips == 0
output = root / 'build/broad-validation-motion-v1'
output.mkdir(exist_ok=False)
vector_file = output / 'vectors.hex'
with vector_file.open('x') as stream:
    for values in inputs:
        hidden, raw = predictor.step(values)
        stream.write(''.join(f'{int(value) & 65535:04x}'
                            for value in [*values, *hidden, *raw]) + '\n')
assert sha256(trial) == trial_hash
assert current_identities() == (sources, versions)
manifest = dict(count=count, hidden_size=96, initial_hidden=[0] * 96,
    numeric=predictor.manifest['numeric'], checkpoint_sha256=selection['checkpoint_sha256'],
    export_manifest_sha256=sha256(export / 'manifest.json'), vectors_sha256=sha256(vector_file),
    expected_cycles=124046, trial_id=2, split='validation', start_sample_index=start,
    source_trial_sha256=trial_hash, prepared_manifest_sha256=sha256(prepared / 'manifest.json'),
    movement_samples=int(movement.sum()), valid_movement_samples=int((movement & valid).sum()),
    input_clipping_count=clips, reference_counters=predictor.counters,
    recipe_sha256=sha256(Path(__file__)), parent_study_selection_sha256=sha256(selection_path),
    scope='RESET then1000old-validation motion inputs; numerical equivalence only. Reset at this window differs from full-trial state; no orientation accuracy claim.')
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
