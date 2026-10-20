from pathlib import Path
import json,hashlib
from host.broad_protocol import parse_response
p=Path('build/broad-post-synth-v1/post-route')
r=json.loads(Path('build/broad-fpga-v1/routed.json').read_text())['modules']['top'];s=json.loads(Path('build/broad-fpga-v1/synth.json').read_text())['modules']['broad_top']
ram=0
for n,c in r['cells'].items():
 if c['type']=='SP':assert c['parameters']==s['cells'][n]['parameters'];ram+=1
assert ram==48
# Independent ALU optimized primitive truth tables from nextpnr optimize_alu_lut.
checks=0
for lut,a_const,b_const,add in [(0x303f,1,None,1),(0xc0cf,1,None,0),(0xc0c0,0,None,1),(0x3030,0,None,0),(0x505a,None,1,1),(0xa0aa,None,1,0),(0xa0aa,None,0,1),(0x505a,None,0,0)]:
 for a in range(2):
  for b in range(2):
   for d in range(2):
    for ci in range(2):
     aa=a if a_const is None else a_const;bb=b if b_const is None else b_const
     orig_s=aa^(bb if add else 1-bb);orig_c=aa
     ss=(lut>>(4+8*d+2*b+a))&1;cc=(lut>>(2*b+a))&1
     assert (ss^ci,ci if ss else cc)==(orig_s^ci,ci if orig_s else orig_c)
     checks+=1
step=json.loads(Path('build/broad-h96-physical-validation-diagnostic-v1.json').read_text())['steps'][0]
responses=[]
for line in p.joinpath('replay.log').read_text().splitlines():
 label,h,ticks=line.split();seq,hidden,raw,cycles=parse_response(bytes.fromhex(h),96)
 responses.append({'request':label,'sequence':seq,'hidden':hidden,'raw':raw,'cycles':cycles,'crc_and_status_valid':True,'ticks':int(ticks.split('=')[1])})
 if label=='request-reset.hex':assert seq==65535 and not any(hidden+raw) and cycles==0
 else:assert seq==0 and cycles==124046
assert len(responses)==2
out=responses[1]
hm=[{'neuron':i,'actual':a,'expected':b,'xor':(a^b)&65535} for i,(a,b) in enumerate(zip(out['hidden'],step['expected_hidden'])) if a!=b]
rm=[{'output':i,'actual':a,'expected':b} for i,(a,b) in enumerate(zip(out['raw'],step['expected_raw'])) if a!=b]
files=[str(x) for x in p.iterdir() if x.is_file() and x.name!='report.json']+['build/broad-fpga-v1/routed.json','build/broad-post-synth-v1/report.json','build/broad-post-synth-v1/cells_logic.v','build/broad-post-synth-v1/rom_models.sv','build/broad-post-synth-v1/replay.cpp']
report={'scope':'Zero-delay post-nextpnr logical cells via strict diagnostic structural adapter, UART RESET then first old-validation02 STEP','result':'PASS: all Golden values exact' if not hm and not rm else 'MISMATCH: retained, not passing equivalence','hidden_mismatches':hm,'raw_mismatches':rm,'all48_ram_configs_and_inits_unchanged':True,'independent_optimized_alu_truth_table_checks':checks,'responses':responses,'limitations':['Adapter covers observed restricted cells and validates disconnected ALU inputs are irrelevant. It is not a vendor gate-level timing simulation.','RAW_ALU_LUT semantics derive from current official nextpnr source snapshot and installed apycula tables. Actual tool revision correspondence is not independently verified.','No bitstream reconstruction, routing delay, PIP correctness or hardware configuration is modeled.'],'files_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(files)}}
p.joinpath('report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['result']);print(hm,rm)
