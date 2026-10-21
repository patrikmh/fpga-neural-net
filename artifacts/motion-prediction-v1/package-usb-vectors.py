"""Package a reset-aligned evaluated motion sequence for USB equivalence only."""

import argparse
import json
from pathlib import Path

import numpy as np

from model.broad_integer import GoldenBroadPredictor, sha256


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export-dir', type=Path, required=True)
parser.add_argument('--evaluation', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
predictor = GoldenBroadPredictor.from_export(args.export_dir)
report = json.loads(args.evaluation.read_text())
if (report['mode'] != 'integer'
        or report['export_manifest_sha256'] != sha256(args.export_dir / 'manifest.json')):
    raise ValueError('Requires integer evaluation of this exact export')
sequence = next(row for row in report['sequences'] if row['family'] == 'changing_axis')
prediction_file = args.evaluation.parent / sequence['predictions_file']
with np.load(prediction_file, allow_pickle=False) as arrays:
    inputs = arrays['quantized_inputs'][:1000].copy()
    states = arrays['hidden'][:1000].copy()
    outputs = arrays['raw_outputs'][:1000].copy()
if (inputs.shape != (1000, 9) or states.shape != (1000, predictor.hidden_size)
        or outputs.shape != (1000, 4)):
    raise ValueError('Requires 1000 reset-aligned input/state/output rows with matching shapes')
lines = []
for values, expected_hidden, expected_raw in zip(inputs, states, outputs):
    hidden, raw = predictor.step(values)
    if not np.array_equal(hidden, expected_hidden) or not np.array_equal(raw, expected_raw):
        raise ValueError('Stored evaluation differs from reset-aligned integer replay')
    lines.append(''.join(f'{int(value) & 65535:04x}' for value in [*values, *hidden, *raw]) + '\n')
args.output_dir.mkdir(parents=True, exist_ok=False)
vectors = args.output_dir / 'vectors.hex'
vectors.write_text(''.join(lines))
manifest = dict(count=1000, hidden_size=predictor.hidden_size,
                initial_hidden=[0] * predictor.hidden_size, numeric=predictor.manifest['numeric'],
                checkpoint_sha256=predictor.manifest['checkpoint_sha256'],
                export_manifest_sha256=sha256(args.export_dir / 'manifest.json'),
                vectors_sha256=sha256(vectors), evaluation_sha256=sha256(args.evaluation),
                prediction_file_sha256=sha256(prediction_file), sequence_id=sequence['sequence_id'],
                split=report['split'], family=sequence['family'], start_sample_index=0,
                recipe_sha256=sha256(Path(__file__)),
                scope='Numerical USB equivalence only; not an accuracy qualification.')
(args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
