"""Compile and run the source-bound H96 10k Verilator scale gate."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path.cwd()
work=root/'build/broad-rtl-v4'
previous=root/'artifacts/broad-rtl-v3/report.json'
previous_sha256='75e3b34c53d9107271ffe7519a1972c0f4c79809150ad791c15c9d0877d49a76'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(previous)==previous_sha256
for name in ('obj','simulation-freeze.json','compile.log','replay.log','report.json'):
    assert not (work/name).exists(), f'Output already exists: {work/name}'
stage=json.loads((work/'stage-manifest.json').read_text())
for entry in stage['banks']:assert sha(work/'rom'/entry['file'])==entry['sha256']
for name,value in stage['activation_sha256'].items():assert sha(work/'rom'/name)==value
assert sha(work/'vectors.hex')==stage['vectors']['sha256']
assert sha(root/'build/broad-integer-v1/export/manifest.json')==stage['export_manifest_sha256']
sources={str(p.relative_to(root)):sha(p) for p in (root/'rtl/broad_core.sv',work/'replay.cpp',Path(__file__).resolve())}
verilator=root/'.tools/tang-toolchain/oss-cad-suite/bin/verilator'
version=subprocess.check_output([str(verilator),'--version'],text=True).strip()
freeze={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage_manifest_sha256':sha(work/'stage-manifest.json'),'sources':sources,'verilator':version,'previous_core_report_sha256':previous_sha256}
with (work/'simulation-freeze.json').open('x') as stream:stream.write(json.dumps(freeze,indent=2)+'\n')
obj=work/'obj';assert not obj.exists()
command=[str(verilator),'--cc','--exe','--build','-j','2','-O3','--top-module','broad_core','--prefix','Vbroad_core','--Mdir',str(obj),'-GHIDDEN_SIZE=96',f'-GWEIGHTS_DIR="{work/"rom"}"',f'-GTANH_FILE="{work/"rom/tanh.hex"}"',f'-GSIGMOID_FILE="{work/"rom/sigmoid.hex"}"',str(root/'rtl/broad_core.sv'),str(work/'replay.cpp')]
with (work/'compile.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
command=[str(obj/'Vbroad_core'),str(work/'vectors.hex'),'10000']
with (work/'replay.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
lines=(work/'replay.log').read_text().splitlines()
result=json.loads(next(line.removeprefix('RESULT ') for line in lines if line.startswith('RESULT ')))
assert result['matched_steps']==10000 and result['reset_replay']
assert result['cycles']==json.loads(previous.read_text())['result']['cycles']
for name,value in sources.items():assert sha(root/name)==value
assert sha(previous)==previous_sha256
assert sha(work/'stage-manifest.json')==freeze['stage_manifest_sha256']
report=dict(freeze,result=result,core_time_us_at_50mhz=result['cycles']/50,physical_verification=False,cycle_schedule_unchanged=True,previous_core_report_sha256=sha(previous),stage_manifest_sha256=sha(work/'stage-manifest.json'),compile_log_sha256=sha(work/'compile.log'),replay_log_sha256=sha(work/'replay.log'),binary_sha256=sha(obj/'Vbroad_core'))
with (work/'report.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
