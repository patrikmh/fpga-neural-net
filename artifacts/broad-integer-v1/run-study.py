"""Run frozen validation-only integer study, independently audit saved traces."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
root=Path.cwd()
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'tests'))
from integration.test_broad_integer import independent_broad_step, full_rmse

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

plan_path=root/'experiments/broad-numeric-study-v1.json'
plan=json.loads(plan_path.read_text())
checkpoint=root/plan['checkpoint']
assert digest(checkpoint)==plan['checkpoint_sha256']
prepared=root/'build/broad-v1-prepared'
output=root/'build/broad-integer-v1'
output.mkdir(exist_ok=False)
sources={name:digest(root/name) for name in ('model/broad_integer.py','model/broad_export_integer.py','model/broad_evaluate_integer.py','tests/integration/test_broad_integer.py','model/broad_train.py','model/broad_network.py','model/broad_evaluate.py')}
freeze={'plan':plan,'plan_sha256':digest(plan_path),'sources':sources,'runner_sha256':digest(Path(__file__)),'prepared_manifest_sha256':digest(prepared/'manifest.json'),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(output/'study-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
exported=output/'export'
evaluated=output/'validation'
commands=[['model.broad_export_integer','--checkpoint',str(checkpoint),'--output-dir',str(exported)],['model.broad_evaluate_integer','--export-dir',str(exported),'--prepared-dir',str(prepared),'--output-dir',str(evaluated)]]
for i,args in enumerate(commands):
    with (output/f'command{i}.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-m',*args],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
manifest=json.loads((exported/'manifest.json').read_text())
def words(name):
    values=[int(w,16) for w in (exported/name).read_text().split()]
    return [v-65536 if v>=32768 else v for v in values]
weights,tanh,sigmoid=(words(name) for name in ('weights.hex','tanh.hex','sigmoid.hex'))
report=json.loads((evaluated/'report.json').read_text())
assert [t['trial_id'] for t in report['trials']]==plan['validation_trials']
scores={'float':[],'integer':[]}
checks=0
for entry in report['trials']:
    with np.load(evaluated/entry['predictions_file'],allow_pickle=False) as saved:
        mask=saved['movement'] & saved['reference_valid'];mask[:32]=False
        for kind in scores:
            score=full_rmse(saved[f'{kind}_predictions'],saved['targets'],mask)
            np.testing.assert_allclose(score,entry[kind]['full_rmse_deg'],rtol=0,atol=1e-9)
            scores[kind].append(score)
        n,h=saved['hidden'].shape
        indexes=np.unique(np.concatenate((np.arange(min(16,n)),np.random.default_rng(731).integers(0,n,size=64))))
        for index in indexes:
            old=saved['hidden'][index-1].tolist() if index else [0]*h
            hidden,raw,*_=independent_broad_step(saved['quantized_inputs'][index].tolist(),old,weights,tanh,sigmoid)
            np.testing.assert_array_equal(hidden,saved['hidden'][index]);np.testing.assert_array_equal(raw,saved['raw_outputs'][index]);checks+=1
for kind,values in scores.items():
    np.testing.assert_allclose(np.mean(values),report['mean_trial_metrics'][kind]['full_rmse_deg'],rtol=0,atol=1e-9)
np.testing.assert_allclose(np.mean(scores['float']),26.433936383466115,rtol=0,atol=1e-9)
for name,value in sources.items(): assert digest(root/name)==value
assert digest(plan_path)==freeze['plan_sha256']
audit={'independent_scalar_checked_steps':checks,'mean_full_rmse_deg':{k:float(np.mean(v)) for k,v in scores.items()},'meets_precision_target':report['meets_precision_target'],'report_sha256':digest(evaluated/'report.json'),'export_manifest_sha256':digest(exported/'manifest.json'),'new_holdout_accessed':False}
(output/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
