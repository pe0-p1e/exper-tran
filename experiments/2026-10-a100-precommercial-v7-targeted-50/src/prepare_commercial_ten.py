"""Freeze a balanced ten-image cohort; never select by commercial predictions."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from proxy_study_protocol import OUT, ROOT, EXP, sha, atomic_json, fingerprint

BASE=OUT/'commercial_pilot10'
DEST=ROOT/'commercial_ten_image'
TIDS=['T01','T03','T05','T07','T09']


def freeze(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError('Never overwrite a changed commercial freeze: '+str(path))
    atomic_json(path,value)


def prepare():
    choice_path=OUT/'ten_hour_research/selected_method_lock.json'
    choice=json.loads(choice_path.read_text())
    if choice['selection']!='held-out surrogate native outputs only' or choice['final_target_feedback']:
        raise RuntimeError('Missing surrogate-only method selection provenance')
    attack=Path(choice['selected']['root'])
    previous=OUT/'commercial_pilot3'
    old=json.loads((previous/'private/manifest.json').read_text())
    old_ids={(c['transition'],c['image_id'],kind):rid for c in old['cases'] for kind,rid in c['requests'].items()}
    prompt=next((previous/'blinded').glob('*/prompt.txt')).read_bytes()
    references=set()
    for p in (OUT/'manifests').rglob('*.jsonl'):
        if 'ref' in str(p.relative_to(OUT/'manifests')).lower():
            references.update(json.loads(s)['image_id'] for s in p.read_text().splitlines() if s.strip())
    if not references:
        raise RuntimeError('Reference manifest unavailable; cannot verify disjointness')
    cases=[];requests=[];selection_hashes={}
    for tid in TIDS:
        dev=OUT/'manifests/dev'/f'{tid}.jsonl'
        selection_hashes[tid]=sha(dev)
        rows=[json.loads(s) for s in dev.read_text().splitlines() if s.strip()][:2]
        testids={json.loads(s)['image_id'] for s in (OUT/'manifests/test'/f'{tid}.jsonl').read_text().splitlines() if s.strip()}
        for row in rows:
            iid=row['image_id']
            if iid in references or iid in testids:raise RuntimeError('Development/reference/test overlap')
            matches=[]
            for p in (attack/tid).glob('batch_*/state.json'):
                d=json.loads(p.read_text())
                if iid in d['image_ids']:matches.append((p,d,d['image_ids'].index(iid)))
            if len(matches)!=1:raise RuntimeError('Missing or duplicate frozen attack: '+iid)
            state,d,i=matches[0]
            if d['target_access'] or d['steps']!=50 or d['seed']!=42 or d['status']!='attack_complete':
                raise RuntimeError('Invalid attack provenance')
            if d['models']!=choice['selected']['config']['models'] or d['source_label']!=int(row['human_label']):
                raise RuntimeError('Method or ground-truth mismatch')
            pair={};pixels=[]
            for kind,key in [('clean','clean_paths'),('adversarial','adv_paths')]:
                src=OUT/d[key][i]
                if sha(src)!=d['png_hashes'][d[key][i]]:raise RuntimeError('Frozen attack PNG changed')
                with Image.open(src) as image:
                    if image.size!=(224,224) or image.info:raise RuntimeError('Canvas/metadata mismatch')
                    pixels.append(np.asarray(image.convert('RGB')).astype('int16'))
                rid=old_ids.get((tid,iid,kind),fingerprint(['commercial-ten-v1',tid,iid,kind])[:16])
                target=BASE/'blinded'/rid;target.mkdir(parents=True,exist_ok=True)
                if (target/'image.png').exists() and sha(target/'image.png')!=sha(src):raise RuntimeError('Changed existing pack PNG')
                if (target/'prompt.txt').exists() and (target/'prompt.txt').read_bytes()!=prompt:raise RuntimeError('Changed existing pack prompt')
                shutil.copyfile(src,target/'image.png');(target/'prompt.txt').write_bytes(prompt)
                pair[kind]=rid
                requests.append(dict(request_id=rid,image_sha256=sha(src),prompt_sha256=sha(target/'prompt.txt')))
            delta=int(np.abs(pixels[0]-pixels[1]).max())
            if delta>16:raise RuntimeError('Perturbation bound exceeded')
            cases.append(dict(transition=tid,image_id=iid,source_label=d['source_label'],target_label=d['target_label'],
                              requests=pair,max_uint8_delta=delta,source_state=str(state.relative_to(OUT)),source_models=d['models'],steps=50,seed=42))
    manifest=dict(protocol='ten_image_frozen_method_cli_v1',cases=cases,requests=sorted(requests,key=lambda r:r['request_id']),
                  selection='First two fixed development IDs in each of five directions; no output filtering',
                  method_lock_sha256=sha(choice_path),dev_manifest_sha256=selection_hashes,
                  primary_denominator=10,goal_hits=6,reasoning_effort='low',
                  previous_pilot_images_included=3,previously_unobserved_images=7,
                  exploratory=True,target_feedback_for_tuning=False)
    freeze(BASE/'private/manifest.json',manifest)
    freeze(BASE/'request_order.json',[r['request_id'] for r in manifest['requests']])
    # This is a separate study; freeze identity before issuing any new request.
    freeze(DEST/'method_and_cohort_lock.json',dict(manifest_sha256=sha(BASE/'private/manifest.json'),
           model='gpt-6.1-sol',client='codex-cli 0.163.0',reasoning_effort='low',
           selection='Previously frozen surrogate-development winner; new composition study has not selected a replacement',
           new_target_calls_max=14,planned_total_requests=20,goal_hits=6,denominator=10,
           feedback_for_tuning=False))
    print('Frozen 10 images / five directions; 3 previously observed, 7 new.',flush=True)
    return manifest


def reuse_previous(manifest):
    """Reuse identical observations, including valid negative predictions."""
    source=ROOT/'commercial_cli/codex/gpt-6.1-sol'
    dest=DEST/'codex/gpt-6.1-sol'
    count=0
    for request in manifest['requests']:
        rid=request['request_id'];record=source/rid/'result.json'
        if not record.exists():continue
        d=json.loads(record.read_text());identity=d['identity']
        if identity!=dict(image_sha256=request['image_sha256'],prompt_sha256=request['prompt_sha256'],client='codex-cli 0.163.0',model='gpt-6.1-sol'):
            raise RuntimeError('Previous observation identity differs')
        if not (dest/rid).exists():
            shutil.copytree(record.parent,dest/rid)
            atomic_json(dest/rid/'reuse_provenance.json',dict(source=str(record),sha256=sha(record),new_inference=False))
        count+=1
    print('Reused',count,'identical prior requests (including failures to hit target).',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');args=p.parse_args()
    manifest=prepare()
    if args.reuse:reuse_previous(manifest)


if __name__=='__main__':main()
