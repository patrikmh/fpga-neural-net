from pathlib import Path
import json,copy,re,hashlib,collections
p=Path('build/broad-post-synth-v1/post-route');orig=json.loads(Path('build/broad-fpga-v1/routed.json').read_text());j=copy.deepcopy(orig);m=j['modules'].pop('top');j['modules']['broad_top']=m;m.pop('settings',None)
allowed={'LUT1','LUT2','LUT3','LUT4','DFFRE','DFFSE','DFF','MUX2_LUT5','MUX2_LUT6','MUX2_LUT7','MUX2_LUT8','IBUF','OBUF','ALU','SP','GOWIN_GND','GOWIN_VCC','PINCFG','GSR'}
audit=[]
for n,c in list(m['cells'].items()):
 typ=c['type'];assert typ in allowed,(n,typ)
 if typ in ['IBUF','OBUF']:
  assert c['parameters']=={'NET_I':'NET','NET_O':'NET'};c['parameters']={};audit.append([n,'NET_I/NET_O routing metadata removed'])
 if typ=='PINCFG':
  assert all(v in ([50108],[50109]) for v in c['connections'].values());assert all(v=='input' for v in c['port_directions'].values());del m['cells'][n];audit.append([n,'input-only pin configuration removed']);continue
 if typ=='GSR':
  assert c['connections']=={'GSRI':[50108]};del m['cells'][n];audit.append([n,'global reset held inactive VCC removed']);continue
 if typ in ['GOWIN_GND','GOWIN_VCC']:
  oldport,newport,newtype=('G','G','GND') if typ=='GOWIN_GND' else ('V','V','VCC');c['type']=newtype;continue
 if typ=='SP':
  w=int(c['parameters']['BIT_WIDTH'],2);assert w in [4,9,18];c['type']='SP' if w==4 else 'SPX9';con=c['connections'];dirs=c['port_directions'];maxdo=32 if w==4 else 36
  for bus,num in [('DO',maxdo),('DI',maxdo),('AD',14),('BLKSEL',3)]:
   bits=[]
   for k in range(num):
    key=bus+str(k);assert key in con,(n,key);assert len(con[key])==1 or (bus=='DI' and not con[key]);bits+=(con.pop(key) or ['x']);dirs.pop(key)
   con[bus]=bits;dirs[bus]='output' if bus=='DO' else 'input'
  assert set(con)=={'DO','DI','AD','BLKSEL','WRE','CLK','CE','OCE','RESET'},set(con)
  audit.append([n,'bus regroup only; params and actual bits preserved']);continue
 if typ=='ALU':
  c['type']='DIAG_ALU';ps=c['parameters'];mode=ps.get('ALU_MODE');raw=ps.get('RAW_ALU_LUT');assert mode in [None,'C2L','00000000000000000000000000000010']
  if not c['connections']:
   assert mode=='C2L';del m['cells'][n];audit.append([n,'portless dummy removed']);continue
  assert c['connections'].get('I2')==[50108]
  assert ps.get('CIN_NETTYPE','WIRE') in ['WIRE','GND','VCC','LOGIC']
  val=int(raw,2) if raw else (0 if mode=='C2L' else 0x609a)
  # All removed input pins must be irrelevant to both internal S and C truth tables.
  def sc(a,b,d):return ((val>>(4+8*d+2*b+a))&1,(val>>(2*b+a))&1)
  for port,idx in [('I0',0),('I1',1),('I3',2)]:
   if c['connections'].get(port):continue
   for x in range(8):
    v=[(x>>b)&1 for b in range(3)];other=v[:];other[idx]^=1;assert sc(*v)==sc(*other),(n,port,hex(val))
  audit.append([n,'raw ALU LUT evaluated, missing inputs proven irrelevant'])
m['cells']={'diag_cell_%d'%i:c for i,c in enumerate(m['cells'].values())}
p.joinpath('adapted.json').write_text(json.dumps(j))
p.joinpath('adapter-audit.json').write_text(json.dumps({'routed_sha256':hashlib.sha256(Path('build/broad-fpga-v1/routed.json').read_bytes()).hexdigest(),'changes':audit,'count':len(audit)},indent=2)+'\n')
