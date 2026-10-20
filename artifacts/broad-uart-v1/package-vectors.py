"""Package the verified H96 RTL replay trace for the USB verifier."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export-dir',type=Path,required=True)
parser.add_argument('--rtl-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
export,rtl,out=args.export_dir,args.rtl_dir,args.output_dir
a=json.loads((export/'manifest.json').read_text());r=json.loads((rtl/'report.json').read_text());stage=json.loads((rtl/'stage-manifest.json').read_text())
assert sha(export/'manifest.json')==stage['export_manifest_sha256']
assert sha(rtl/'stage-manifest.json')==r['stage_manifest_sha256']
assert sha(rtl/'vectors.hex')==stage['vectors']['sha256']
assert r['result']['matched_steps']==stage['vectors']['steps']==10000 and r['result']['reset_replay']
assert stage['hidden_size']==a['config']['hidden_size']==96
out.mkdir(exist_ok=False)
shutil.copy2(rtl/'vectors.hex',out/'vectors.hex')
m={'count':10000,'hidden_size':96,'initial_hidden':[0]*96,'numeric':a['numeric'],'export_manifest_sha256':sha(export/'manifest.json'),'checkpoint_sha256':a['checkpoint_sha256'],'vectors_sha256':sha(out/'vectors.hex'),'expected_cycles':r['result']['cycles'],'rtl_report_sha256':sha(rtl/'report.json'),'rtl_core_sha256':r['sources']['rtl/broad_core.sv'],'source_stage_manifest_sha256':sha(rtl/'stage-manifest.json')}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
