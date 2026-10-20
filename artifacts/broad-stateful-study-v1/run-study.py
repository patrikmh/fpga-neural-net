"""Run the predeclared two-seed chronological-training comparison."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.broad_evaluate import orientation_metrics
from model.broad_train import sha256
from model.broad_train_stateful import current_provenance

root = Path.cwd()
plan_path = root / 'experiments/broad-stateful-study-v1.json'
plan = json.loads(plan_path.read_text())
initial = root / plan['initial_checkpoint']
prepared = root / 'build/broad-v1-prepared'
assert sha256(initial) == plan['initial_checkpoint_sha256']
assert sha256(prepared / 'manifest.json') == plan['training_prepared_manifest_sha256']
output = root / 'build/broad-stateful-v1'
output.mkdir(exist_ok=False)
provenance = current_provenance()
recipe = {'model/broad_train_stateful.py': sha256(root / 'model/broad_train_stateful.py')}
freeze = {'plan': plan, 'plan_sha256': sha256(plan_path), 'provenance': provenance,
          'training_recipe_sources': recipe, 'script_sha256': sha256(Path(__file__)),
          'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (output / 'study-freeze.json').open('x') as stream:
    json.dump(freeze, stream, indent=2, allow_nan=False)
steps_per_pass = 0
for path in sorted((prepared / 'train').glob('*.npz')):
    with np.load(path, allow_pickle=False) as trial:
        mask = (trial['movement'] & trial['reference_valid']).copy()
        mask[:plan['warmup_samples_per_trial']] = False
        steps_per_pass += sum(bool(mask[start:start + plan['chunk_size']].any())
                              for start in range(0, len(mask), plan['chunk_size']))
results = []
for seed in plan['seeds']:
    assert current_provenance() == provenance
    assert recipe['model/broad_train_stateful.py'] == sha256(root / 'model/broad_train_stateful.py')
    destination = output / f'seed{seed}'
    command = [str(root / '.venv/bin/python'), '-m', 'model.broad_train_stateful',
               '--prepared-dir', str(prepared), '--initial-checkpoint', str(initial),
               '--initial-checkpoint-sha256', plan['initial_checkpoint_sha256'],
               '--output-dir', str(destination), '--epochs', str(plan['epochs_per_run']),
               '--chunk-size', str(plan['chunk_size']), '--learning-rate', str(plan['learning_rate']),
               '--seed', str(seed)]
    print(f'START stateful seed={seed}', flush=True)
    with (output / f'seed{seed}.log').open('x') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((destination / 'report.json').read_text())
    assert report['optimizer_steps'] == steps_per_pass * plan['epochs_per_run']
    np.testing.assert_allclose(report['initial_validation_rmse_deg'], plan['baseline_validation_rmse_deg'], rtol=0, atol=1e-9)
    per_trial = []
    for entry in report['validation']:
        with np.load(destination / entry['predictions_file'], allow_pickle=False) as trial:
            mask = (trial['movement'] & trial['reference_valid']).copy()
            mask[:plan['warmup_samples_per_trial']] = False
            metrics = orientation_metrics(trial['predictions'], trial['targets'], mask, 1e-12)
        per_trial.append({'trial_id': entry['trial_id'], **metrics})
    means = {key: float(np.mean([trial[key] for trial in per_trial]))
             for key in ('full_rmse_deg', 'inclination_rmse_deg', 'heading_rmse_deg')}
    np.testing.assert_allclose(means['full_rmse_deg'], report['validation_rmse_deg'], rtol=0, atol=1e-9)
    result = {'seed': seed, 'best_epoch': report['best_epoch'], 'optimizer_steps': report['optimizer_steps'],
              'best_optimizer_steps': report['best_optimizer_steps'], 'parameter_count': report['parameter_count'],
              'metrics': means, 'trials': per_trial,
              'checkpoint': str((destination / 'checkpoint.pt').relative_to(root)),
              'checkpoint_sha256': sha256(destination / 'checkpoint.pt'),
              'training_report_sha256': sha256(destination / 'report.json')}
    results.append(result)
    with (output / 'results.jsonl').open('a') as stream:
        stream.write(json.dumps(result, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)
selected = min(results, key=lambda result: (result['metrics']['full_rmse_deg'], result['seed']))
improved = selected['metrics']['full_rmse_deg'] < plan['baseline_validation_rmse_deg']
summary = {'freeze': freeze, 'results': results, 'improved': improved,
           'selected_checkpoint': selected['checkpoint'] if improved else plan['initial_checkpoint'],
           'selected': selected, 'steps_per_pass': steps_per_pass,
           'new_holdout_accessed': False, 'physical_fit': 'not yet established'}
with (output / 'summary.json').open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
print('STATEFUL COMPLETE: ' + json.dumps({'improved': improved, 'selected_checkpoint': summary['selected_checkpoint'],
                                        'metrics': selected['metrics']}), flush=True)
