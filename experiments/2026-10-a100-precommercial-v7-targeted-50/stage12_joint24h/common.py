"""V8 study state: no inference or external requests at import."""
import hashlib,json,os,sys,tempfile,time,fcntl
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
REPO=EXP.parents[1]
MAIN=Path(os.environ.get('V8_MAIN_REPO','/home/ubuntu/exper-tran'))
DATA=MAIN/'outputs/a100_precommercial_v7_targeted50'
ROOT=Path(os.environ.get('V8_OUTPUT_ROOT',str(MAIN/'outputs/a100_precommercial_v8_joint_selection24h')))
os.environ['V7_DATA_ROOT']=str(DATA)
sys.path[:0]=[str(EXP/'src'),str(MAIN/'src')]
import proxy_study_protocol
from supplement_registry import registry
REG=registry()
VALIDATORS={'qwen':'Qwen3.5-9B','internvl':'InternVL3.5-8B-HF','gemma4':'Gemma 4 E4B-it'}
LARGE_EVAL_MODES={'evaluate_large_qwen':'Qwen3.5-27B','evaluate_large_internvl':'InternVL3.5-14B-HF','evaluate_large_gemma4':'Gemma 4 31B-it'}
V='CLIP ViT-L/14';D='DINOv2-Large';Q='Qwen3.5-2B';I='InternVL3.5-2B-HF';G='Gemma 4 E2B-it'
POOLS={'qwen':[V,I,G,D],'internvl':[V,Q,G,D],'gemma4':[V,Q,I,D]}
TIDS=['T01','T03','T05','T07','T09']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
  d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def freeze(p,x):
 if Path(p).exists() and read(p)!=x:raise RuntimeError('Immutable artifact differs: '+str(p))
 if not Path(p).exists():atomic(p,x)
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def native(n,fracs):
 import math
 d=REG[n]['expected_vision_depth'];return sorted({max(0,min(d-1,math.ceil(f*d)-1)) for f in fracs})
def legal(layers,validator):
 if not layers:raise ValueError('Empty proxy set')
 fam=REG[validator]['family'] if validator in REG else validator
 for n,ls in layers.items():
  if n not in REG or REG[n]['family']==fam:raise ValueError('Target family collision: '+n)
  if not ls or len(ls)!=len(set(ls)) or any(type(l)!=int or not 0<=l<REG[n]['expected_vision_depth'] for l in ls):raise ValueError('Invalid native layers')
 return True
def make(family,layers,split='screen',label='candidate',smoke=False):
 legal(layers,VALIDATORS[family]);layers={n:sorted(ls) for n,ls in layers.items()}
 c=dict(version='8.1',model_order=list(layers),family=family,validator=VALIDATORS[family],layers=layers,split=split,steps=2 if smoke else 50,seed=42,
  epsilon=16/255,alpha=1/255,momentum=1.,pull=.5,push=.5,DI=True,CLS=0,patch=0,parameter_dtype='bfloat16',
  fusion='equal_per_image_L1_proxy_gradient_mean',layer_reduction='mean_before_proxy_normalization',
  RNG='sha256(seed,image_id,stream); individual forward per image independent of microbatch')
 c['id']='V8_'+digest(c)[:18];c['label']=label
 return c
def scientific(c):return {k:v for k,v in c.items() if k not in ['id','label']}
def cohort(split):return read(ROOT/'manifests'/f'{split}.json')
def image_receipt(c,iid):return ROOT/'attacks'/c['id']/digest(iid)[:18]/'state.json'
def valid_receipt(p,c=None):
 d=read(p)
 if not d or d.get('status')!='COMPLETE':return False
 if c and d['science']!=digest(scientific(c)):raise RuntimeError('Changed attack condition')
 for path,h in d.get('png_hashes',{}).items():
  if not Path(path).exists() or sha(path)!=h:return False
 return bool(d.get('png_hashes')) and d.get('linf_png',1)<=16/255+1e-7 and not d.get('target_access',True)
def attack_done(c):return all(valid_receipt(image_receipt(c,r['image_id']),c) for r in cohort(c['split'])['images'])
def eval_path(c):return ROOT/'validation'/(c['id']+'.json')
def eval_done(c):
 d=read(eval_path(c));return bool(d and d.get('complete') and d.get('science')==digest(scientific(c)) and d['N']==len(cohort(c['split'])['images']))
def seed(iid,stream):return int(digest([42,iid,stream])[:15],16)
def proc_identity(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();after=s[s.rfind(')')+2:].split()
  return dict(pid=pid,start=after[19],command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' '),state=after[0])
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def same_process(old):
 cur=proc_identity(old['pid']);return bool(cur and cur['state']!='Z' and cur['start']==old['start'] and cur['command']==old['command'])
def source_hashes():
 files=list(HERE.glob('*.py'))+[HERE/'protocol.yaml']
 files+=list((HERE/'paper_closeout').glob('*.py'))
 files += [EXP/'src'/n for n in ['vision_features.py','pp4_objective.py','representation_objectives.py','attack_optimizers.py','supplement_registry.py','run_codex_cli_pilot.py','family_protocol.py']]
 return {str(p.relative_to(REPO)):sha(p) for p in files}

def effective_deadline():
 extension=read(ROOT/'closeout_window.json')
 return extension['deadline'] if extension else read(ROOT/'window.json',{}).get('deadline',time.time())

def timebox_enabled():
 return not read(ROOT/'run_policy.json',{}).get('run_until_complete',False)

def deadline_reached(deadline,reserve_seconds=0):
 return timebox_enabled() and time.time()>=deadline-reserve_seconds
def check_lock():
 lock=read(ROOT/'study_lock.json')
 if not lock:raise RuntimeError('Missing study lock')
 if lock['sources']!=source_hashes():raise RuntimeError('Scientific/controller source lock mismatch')
 for name,h in lock['manifests'].items():
  if sha(ROOT/'manifests'/name)!=h:raise RuntimeError('Frozen manifest mismatch')
 for path,h in lock.get('input_hashes',{}).items():
  if sha(path)!=h:raise RuntimeError('Input/reference PNG checksum changed')
 return lock

def await_identity(pid,expected,timeout=5):
 deadline=time.time()+timeout
 while time.time()<deadline:
  identity=proc_identity(pid)
  if identity and identity['state']!='Z' and expected in identity['command']:return identity
  time.sleep(.01)
 raise RuntimeError('Child did not expose expected executable identity; inspect retained launch log')
