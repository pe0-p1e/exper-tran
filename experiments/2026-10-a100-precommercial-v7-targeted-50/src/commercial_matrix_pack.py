"""Frozen-package provenance, without model inference or prediction screening."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

from proxy_study_protocol import OUT, ROOT, sha, fingerprint, atomic_json
from prepare_commercial_ten import prepare as prepare_ten
from run_codex_cli_pilot import checked_order

MATRIX=ROOT/'commercial_model_versions'
STAGES={'smoke3':3,'dev10':10,'test150':150}


def freeze(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError('Frozen commercial plan changed: '+str(path))
    atomic_json(path,value)


def full_pack():
    base=OUT/'commercial_test150'
    choicefile=OUT/'ten_hour_research/selected_method_lock.json'
    choice=json.loads(choicefile.read_text())
    if choice['final_target_feedback'] or choice['selection']!='held-out surrogate native outputs only':
        raise RuntimeError('Missing surrogate-only selection provenance')
    attack=OUT/'attacks/test'/choice['allowed_test_conditions'][0]
    template=next((OUT/'commercial_pilot3/blinded').glob('*/prompt.txt')).read_bytes()
    cases=[];requests=[];cohorts={}
    for tid in ['T01','T03','T05','T07','T09']:
        manifest=OUT/'manifests/test'/f'{tid}.jsonl';cohorts[tid]=sha(manifest)
        if cohorts[tid]!=choice['test_manifest_hashes'][tid]:raise RuntimeError('Test cohort changed since method selection')
        rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        if len(rows)!=30:raise RuntimeError('Expected fixed30 images per direction')
        states={}
        for p in (attack/tid).glob('batch_*/state.json'):
            d=json.loads(p.read_text())
            for i,iid in enumerate(d['image_ids']):
                if iid in states:raise RuntimeError('Duplicate frozen attack image')
                states[iid]=(p,d,i)
        for row in rows:
            iid=row['image_id']
            if iid not in states:raise RuntimeError('Frozen150-image method incomplete: '+iid)
            p,d,i=states[iid];pair={}
            if d['source_label']!=int(row['human_label']):raise RuntimeError('Ground truth differs')
            for condition,key in [('clean','clean_paths'),('adversarial','adv_paths')]:
                rid=fingerprint(['commercial-fixed150-v1',tid,iid,condition])[:20]
                src=OUT/d[key][i];dest=base/'blinded'/rid;dest.mkdir(parents=True,exist_ok=True)
                if sha(src)!=d['png_hashes'][d[key][i]]:raise RuntimeError('Changed attack PNG')
                if (dest/'image.png').exists() and sha(dest/'image.png')!=sha(src):raise RuntimeError('Changed package PNG')
                if (dest/'prompt.txt').exists() and (dest/'prompt.txt').read_bytes()!=template:raise RuntimeError('Changed package prompt')
                shutil.copyfile(src,dest/'image.png');(dest/'prompt.txt').write_bytes(template)
                requests.append(dict(request_id=rid,image_sha256=sha(src),prompt_sha256=sha(dest/'prompt.txt')))
                pair[condition]=rid
            cases.append(dict(transition=tid,image_id=iid,source_label=d['source_label'],target_label=d['target_label'],
                              requests=pair,source_state=str(p.relative_to(OUT))))
    secret=dict(protocol='commercial_test150_frozen_method_v1',cases=cases,requests=sorted(requests,key=lambda x:x['request_id']),
                method_lock_sha256=sha(choicefile),manifest_hashes=cohorts,
                selection='Entire original GT-only frozen test cohort; no prediction filtering',target_feedback_for_tuning=False)
    freeze(base/'private/manifest.json',secret)
    freeze(base/'request_order.json',[x['request_id'] for x in secret['requests']])
    return base


def prepare_pack(stage):
    if stage=='dev10':prepare_ten();base=OUT/'commercial_pilot10'
    elif stage=='smoke3':base=OUT/'commercial_pilot3'
    elif stage=='test150':base=full_pack()
    else:raise ValueError(stage)
    n=STAGES[stage];order=checked_order(base,n)
    secret=json.loads((base/'private/manifest.json').read_text())
    declared={r['request_id']:r for r in secret['requests']}
    ref_ids=set()
    for p in (OUT/'manifests').rglob('*.jsonl'):
        if 'ref' in str(p.relative_to(OUT/'manifests')).lower():
            ref_ids.update(json.loads(s)['image_id'] for s in p.read_text().splitlines() if s.strip())
    if len(ref_ids)!=480:raise RuntimeError('Expected480 distinct class reference IDs')
    for c in secret['cases']:
        if c['image_id'] in ref_ids:raise RuntimeError('Reference/attack overlap')
        state=json.loads((OUT/c['source_state']).read_text())
        if state['target_access'] or state['steps']!=50 or state['seed']!=42 or state['status']!='attack_complete':
            raise RuntimeError('Invalid frozen attack provenance')
        if state['models']!=['CLIP ViT-L/14','Qwen3.5-2B','InternVL3.5-2B-HF'] or state['method']!='A3':
            raise RuntimeError('Different method or proxy composition')
        config=state['config']
        if (config['layers']!=[.25,.5,.75,1.] or config['patch_weight']!=.35
                or not config['input_diversity'] or config['fusion']!='equal'):
            raise RuntimeError('Different feature/optimization settings')
        i=state['image_ids'].index(c['image_id']);arrays=[]
        for kind,key in [('clean','clean_paths'),('adversarial','adv_paths')]:
            rid=c['requests'][kind];p=base/'blinded'/rid/'image.png';source=OUT/state[key][i]
            if sha(p)!=sha(source) or sha(p)!=state['png_hashes'][state[key][i]]:raise RuntimeError('PNG provenance mismatch')
            with Image.open(p) as im:
                if im.size!=(224,224) or im.info:raise RuntimeError('Unexpected canvas/metadata')
                arrays.append(np.asarray(im.convert('RGB')).astype('int16'))
        if np.abs(arrays[0]-arrays[1]).max()>16:raise RuntimeError('Perturbation exceeds16/255')
    if len({r['prompt_sha256'] for r in declared.values()})!=1:raise RuntimeError('Different prompts')
    return base,order,secret,declared
