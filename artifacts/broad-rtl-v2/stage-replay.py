"""Reconstruct the hash-bound H96 ROM banks and 10k continuous replay input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export-dir',type=Path,required=True)
parser.add_argument('--vectors-file',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
source,out,vector=args.export_dir,args.output_dir,args.vectors_file
manifest=json.loads((source/'manifest.json').read_text())
assert manifest['config']['hidden_size']==96
for name,value in manifest['files'].items():assert sha(source/name)==value
words=(source/'weights.hex').read_text().split();assert len(words)==41092
with vector.open() as stream:rows=[next(stream) for _ in range(10000)]
assert all(len(row.strip())==436 for row in rows)
out.mkdir(exist_ok=False);banks=out/'rom';banks.mkdir()
entries=[];restored=[]
for i,start in enumerate(range(0,len(words),1024)):
    part=words[start:start+1024];valid=len(part);restored.extend(part);part+=['0000']*(1024-valid)
    p=banks/f'weights_{i:03d}.hex';p.write_text(''.join(w+'\n' for w in part))
    entries.append({'bank':i,'file':p.name,'valid_words':valid,'padding_words':1024-valid,'sha256':sha(p)})
assert restored==words
for name in ('tanh.hex','sigmoid.hex'):shutil.copy2(source/name,banks/name)
v=out/'vectors.hex';v.write_text(''.join(rows))
stage={'export_manifest_sha256':sha(source/'manifest.json'),'flat_weights_sha256':sha(source/'weights.hex'),'hidden_size':96,'bank_depth':1024,'banks':entries,'activation_sha256':{name:sha(banks/name) for name in ('tanh.hex','sigmoid.hex')},'vectors':{'file':v.name,'sha256':sha(v),'steps':10000,'source_sha256':sha(vector),'source_trial':2,'order':'nineinputs,Hnewhidden,fourraw;lane0MSB16intext'},'cycle_count':'not yet measured'}
(out/'stage-manifest.json').write_text(json.dumps(stage,indent=2)+'\n')
