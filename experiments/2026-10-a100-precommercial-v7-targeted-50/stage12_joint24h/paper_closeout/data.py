"""New150 test cohort: ground-truth only, ID AND decoded-pixel disjointness."""
import common as C
OUT=C.ROOT/'paper_closeout'

def prepare():
    path=C.ROOT/'manifests/final150.json'
    if path.exists():
        d=C.read(path)
        if len(d['images'])!=150 or any(sum(x['transition']==t for x in d['images'])!=30 for t in C.TIDS):raise RuntimeError('Wrong final150 cohort')
        for r in d['images']:
            if C.sha(r['path'])!=r['image_sha256']:raise RuntimeError('Final150 input changed')
        return d
    import hashlib,numpy as np,torch
    from PIL import Image
    from primary_ml_cka.data.preprocessing import ensure_canvas
    from primary_ml_cka.artifacts.png import save_png_tensor
    forbidden=set();pixels=set();sources={}
    def add_png(p):
        with Image.open(p) as im:pixels.add(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
    bases=[C.DATA/'manifests',C.MAIN/'outputs/a100_precommercial_v6/evaluation/manifests',C.MAIN/'outputs/a100_precommercial_v6_multi_repr_recovery/manifests']
    for base in bases:
        for p in base.rglob('*.jsonl'):
            sources[str(p)]=C.sha(p)
            forbidden.update(x['image_id'] for x in C.rows(p) if 'image_id' in x)
    for split in ['screen','explore','confirm']:
        for r in C.cohort(split)['images']:forbidden.add(r['image_id']);add_png(r['path'])
    for base in [C.DATA/'canonical_images',C.MAIN/'outputs/a100_precommercial_v6/canonical_images']:
        for p in base.rglob('*.png'):add_png(p)
    result=[]
    for t in C.TIDS:
        sample=next(x for x in C.cohort('confirm')['images'] if x['transition']==t);syn=sample['synset'];chosen=[]
        raw=C.MAIN/'data/imagenet_diverse10_minimal/train'/syn
        for p in sorted(raw.glob('*'),key=lambda p:C.digest(['final150',42,syn,p.name])):
            iid='train/'+syn+'/'+p.name
            if iid in forbidden or p.suffix.lower() not in ['.jpeg','.jpg','.png']:continue
            try:
                with Image.open(p) as im:arr=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
                arr=ensure_canvas(arr[None],224)[0].mul(255).round().div(255)
            except (OSError,ValueError):continue
            dest=OUT/'canonical_final150'/syn/(p.stem+'.png');save_png_tensor(arr,dest)
            with Image.open(dest) as im:h=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if h in pixels:dest.unlink();continue
            pixels.add(h);forbidden.add(iid)
            chosen.append(dict(image_id=iid,path=str(dest),image_sha256=C.sha(dest),transition=t,
              source=sample['source'],target=sample['target'],synset=syn))
            if len(chosen)==30:break
        if len(chosen)!=30:raise RuntimeError(f'DATA_BLOCKER {syn}: {len(chosen)}/30 unused images; no reuse or shrink')
        result+=chosen
    d=dict(images=result,seed=42,N=150,selection='GT only deterministic hash ordering',used_for_selection=False,
      ID_and_decoded_pixel_disjoint=True,excluded_manifest_hashes=sources,untracked_external_cohorts_audit_exhaustive=False)
    C.freeze(path,d);C.freeze(OUT/'final150_cohort_audit.json',dict(N=150,manifest_sha256=C.sha(path),reference_attack_disjoint=True,ID_and_decoded_pixel_disjoint=True))
    return d
