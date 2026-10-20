from pathlib import Path
import hashlib,json,re
from host.broad_protocol import parse_response
p=Path('build/broad-post-synth-v1');root=Path('.')
m=json.loads(Path('build/broad-fpga-v1/synth.json').read_text())['modules']['broad_top']
def decode(n):
 c=m['cells'][n];ps=c['parameters'];width=int(ps['BIT_WIDTH'],2);size=256 if c['type']=='SP' else 288
 content=sum(int(ps.get('INIT_RAM_%02X'%i,'0'),2)<<(size*i) for i in range(64))
 return [(content>>(width*i))&((1<<width)-1) for i in range(size*64//width)]
tanh=[int(x,16) for x in Path('artifacts/broad-integer-v1/export/tanh.hex').read_text().split()]
a=[decode('core.tanh_rom.0.%d'%j) for j in range(4)]
assert [sum(a[j][i]<<(4*j) for j in range(4)) for i in range(4096)]==tanh
sig=[int(x,16) for x in Path('artifacts/broad-integer-v1/export/sigmoid.hex').read_text().split()]
a=[decode('core.sigmoid_rom.0.%d'%j) for j in range(3)]
combined=[sum(a[j][i]<<(9*j) for j in range(3)) for i in range(2048)]
assert [v&8191 for v in combined]+[(v>>13)&8191 for v in combined]==sig
s=json.loads(Path('build/broad-h96-physical-validation-diagnostic-v1.json').read_text())['steps'][0]
responses=[]
for line in p.joinpath('replay.log').read_text().splitlines():
 label,h,ticks=line.split();seq,hidden,raw,cycles=parse_response(bytes.fromhex(h),96)
 if label=='request-reset.hex':assert seq==65535 and not any(hidden+raw) and cycles==0
 else:assert seq==0 and list(hidden)==s['expected_hidden'] and list(raw)==s['expected_raw'] and cycles==124046
 responses.append({'request':label,'sequence':seq,'hidden':hidden,'raw':raw,'cycles':cycles,'crc_and_status_valid':True,'ticks':int(ticks.split('=')[1])})
files=['build/broad-fpga-v1/synth.json','.tools/tang-toolchain/oss-cad-suite/share/yosys/gowin/cells_sim.v','.tools/tang-toolchain/oss-cad-suite/share/yosys/gowin/brams_map_gw5a.v','build/broad-h96-physical-validation-diagnostic-v1.json','artifacts/broad-integer-v1/export/manifest.json']
files += [str(f) for f in p.iterdir() if f.is_file() and f.name!='report.json']
report={'scope':'Zero-delay frozen mapped H96 top, UART RESET then first old-validation02 STEP; no physical I/O or new heldout access','result':'PASS: mapped logic first step matches Golden exactly; physical mismatch is not reproduced','limitations':['Restricted custom ROM models support only observed ROM configurations. All INIT values decoded independently and audited against frozen export.','Zero-delay simulation does not prove physical routed/packed configuration correctness or hardware timing.','One diagnostic step only; not a replacement for required physical equivalence.'], 'rom_audit':{'weight_banks':41,'weight_words':41984,'tanh_words':4096,'sigmoid_words':4096,'all_exact':True,'sigmoid_layout':'2048 words of 26 bits split across 3 width9 primitives, low13 is address[11]=0, next13 is address[11]=1'},'responses':responses,'files_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(files)}}
p.joinpath('report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['result'])
