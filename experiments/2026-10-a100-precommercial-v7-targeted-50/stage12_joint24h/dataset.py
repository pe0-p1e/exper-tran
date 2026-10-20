"""Frozen GT-only cohorts, conservative historical-ID and pixel-hash exclusion."""
from common import *
def prepare():
 if (ROOT/'manifests/dataset_audit.json').exists():
  for split in ['screen','explore','confirm']:assert len(cohort(split)['images'])=={'screen':10,'explore':30,'confirm':50}[split]
  return read(ROOT/'manifests/dataset_audit.json')
 from PIL import Image
 import numpy as np,torch
 from primary_ml_cka.data.preprocessing import ensure_canvas
 from primary_ml_cka.artifacts.png import save_png_tensor
 forbidden=set();refids=set();oldhashes=set();excluded_manifests={}
 bases=[MAIN/'outputs/a100_precommercial_v6/evaluation/manifests',DATA/'manifests',MAIN/'outputs/a100_precommercial_v6_multi_repr_recovery/manifests']
 for base in bases:
  for p in base.rglob('*.jsonl'):
   rr=rows(p);forbidden.update(r['image_id'] for r in rr if 'image_id' in r);excluded_manifests[str(p)]=sha(p)
 refs=[]
 for p in sorted((DATA/'manifests/class_references').glob('*.jsonl')):
  rr=rows(p);assert len(rr)==48;refs+=rr;refids.update(r['image_id'] for r in rr)
 for r in refs:
  p=DATA/'canonical_images'/r['relative_path'];oldhashes.add(sha(p))
 # All historical canonical cohort pixels, including cases renamed by old attacks.
 for base in [DATA/'canonical_images',MAIN/'outputs/a100_precommercial_v6/canonical_images']:
  for p in base.rglob('*.png'):oldhashes.add(sha(p))
 screen=[r for t in TIDS for r in rows(DATA/'manifests/dev'/f'{t}.jsonl')[2:4]]
 result={'screen':[], 'explore':[], 'confirm':[]};used=set();newhashes=set()
 for tid in TIDS:
  sample=next(r for r in screen if r['human_label']==int(tid[1:]));syn=sample['synset'];label=int(sample['human_label'])
  for r in [x for x in screen if x['human_label']==label]:
   result['screen'].append({**r,'path':str(DATA/'canonical_images'/r['relative_path']),'transition':tid,'source':label,'target':label+1})
  raw=MAIN/'data/imagenet_diverse10_minimal/train'/syn
  candidates=sorted(raw.glob('*'),key=lambda p:digest(['v8-independent',42,syn,p.name]))
  chosen=[]
  for path in candidates:
   if path.suffix.lower() not in ['.jpeg','.jpg','.png']:continue
   iid=f'train/{syn}/{path.name}'
   if iid in forbidden or iid in used:continue
   try:
    with Image.open(path) as im:a=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
    a=ensure_canvas(a[None],224)[0].mul(255).round().div(255)
   except Exception:continue
   dest=ROOT/'canonical_images'/syn/(path.stem+'.png');save_png_tensor(a,dest);h=sha(dest)
   if h in oldhashes or h in newhashes:dest.unlink();continue
   newhashes.add(h);used.add(iid)
   chosen.append(dict(image_id=iid,path=str(dest),image_sha256=h,transition=tid,source=label,target=label+1,synset=syn,human_label=label))
   if len(chosen)==16:break
  if len(chosen)<16:raise RuntimeError(f'DATA_BLOCKER: {syn} only {len(chosen)}/16 unused images; expand existing ImageNet downloader, never reuse test data')
  result['explore']+=chosen[:6];result['confirm']+=chosen[6:]
 for split,rr in result.items():freeze(ROOT/'manifests'/f'{split}.json',dict(images=rr,seed=42,historical=split=='screen',used_for_selection=split!='confirm'))
 assert not {r['image_id'] for r in result['confirm']}&{r['image_id'] for r in result['explore']}
 audit=dict(status='PASS',reference_count=len(refs),counts={k:len(v) for k,v in result.items()},independent_exclusion_ids=len(forbidden),
  source_manifest_hashes=excluded_manifests,reference_manifest_hashes={p.name:sha(p) for p in (DATA/'manifests/class_references').glob('*.jsonl')},
  pixel_hash_exclusion=True,stage2_uses_confirmation_cohort_after_method_lock=True,stage3_150_test_sealed=True,
  all_historical_outputs_exhaustive=False,caveat='Conservative V6 candidates, V6 recovery and all V7 dev/test/reference IDs plus canonical pixel hashes; any additional untracked external cohort cannot be audited.')
 freeze(ROOT/'manifests/dataset_audit.json',audit);return audit
if __name__=='__main__':print(json.dumps(prepare(),indent=2))
