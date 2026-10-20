from common import *
import subprocess,shutil,platform

def environment():
 from resources import telemetry
 status={}
 for label,cmd in [('branch',['git','branch','--show-current']),('commit',['git','rev-parse','HEAD']),('working_tree',['git','status','--short']),('worktrees',['git','worktree','list'])]:
  status[label]=subprocess.check_output(cmd,cwd=REPO,text=True).strip()
 disk=shutil.disk_usage(ROOT if ROOT.exists() else MAIN)
 status.update(at=time.time(),hostname=platform.node(),python=sys.version,gpu=telemetry(),disk_free_gib=disk.free/2**30,
  permanent_output=str(ROOT.resolve()),root_device=os.stat(MAIN).st_dev,output_device=os.stat(ROOT).st_dev,
  CLI='reuse existing tested subscription chain; no login/connectivity requests during preflight',model='gpt-6.1-sol',effort='low')
 if disk.free<8*2**30:raise RuntimeError('Persistent output disk free below8GiB')
 if status['output_device']!=status['root_device']:raise RuntimeError('Outputs must be on persistent root volume')
 model=next((m for m in read(Path.home()/'.codex/models_cache.json',{}).get('models',[]) if m.get('slug')=='gpt-6.1-sol'),None)
 status['CLI_catalog_image_low']=bool(model and 'image' in model.get('input_modalities',[]) and any(x.get('effort')=='low' for x in model.get('supported_reasoning_levels',[])))
 if not status['CLI_catalog_image_low']:status['stage2_blocker']='BLOCKED_AUTH_OR_MODEL; no fallback'
 atomic(ROOT/'preflight/environment.json',status);return status

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 from dataset import prepare as data
 env=environment();audit=data()
 from PIL import Image
 def pixels(path):
  with Image.open(path) as im:return __import__('hashlib').sha256(im.convert('RGB').tobytes()).hexdigest()
 prior=set()
 for base in [DATA/'canonical_images',MAIN/'outputs/a100_precommercial_v6/canonical_images']:
  for p in base.rglob('*.png'):prior.add(pixels(p))
 new=set();ids=set();input_hashes={}
 for split in ['explore','confirm']:
  for r in cohort(split)['images']:
   if sha(r['path'])!=r['image_sha256']:raise RuntimeError('Input PNG checksum mismatch')
   h=pixels(r['path'])
   if h in prior or h in new or r['image_id'] in ids:raise RuntimeError('Independent cohort pixel/ID overlap')
   new.add(h);ids.add(r['image_id']);input_hashes[r['path']]=sha(r['path'])
 for r in cohort('screen')['images']:input_hashes[r['path']]=sha(r['path'])
 for p in (DATA/'manifests/class_references').glob('*.jsonl'):
  for r in rows(p):
   ip=DATA/'canonical_images'/r['relative_path'];input_hashes[str(ip)]=sha(ip)
 atomic(ROOT/'preflight/input_integrity.json',dict(independent_ID_and_decoded_pixel_hash_pass=True,input_hashes=input_hashes))
 # A smoke cohort has two historical images and never enters selection/confirmation scores.
 smoke={**cohort('screen'),'images':cohort('screen')['images'][:2],'smoke_only':True}
 freeze(ROOT/'manifests/smoke.json',smoke)
 from selection import proxy_conditions,layer_conditions
 for fam in VALIDATORS:
  for c in proxy_conditions(fam)+layer_conditions(fam):legal(c['layers'],c['validator'])
 return dict(status='PASS_WITH_CLI_QUOTA_UNKNOWN',environment=env,dataset=audit)

def lock():
 result=prepare()
 freeze(ROOT/'study_lock.json',dict(version=8,registry=REG,sources=source_hashes(),protocol=sha(HERE/'protocol.yaml'),
  manifests={p.name:sha(p) for p in (ROOT/'manifests').glob('*.json')},
  instruction='User authorized immediate start; Stage1 transitions automatically to Stage2 after immutable method freeze; no paid API fallback',
  scientific='balanced pure PP, strict BF16,50steps,DI,equal normalized gradients; per-image RNG',
  input_hashes=read(ROOT/'preflight/input_integrity.json')['input_hashes'],no_final_target_selection=True,stage3_sealed=True))
 return result
if __name__=='__main__':print(json.dumps(prepare(),indent=2))
