"""Freeze new-class dictionaries and support arbitrary-C fixed-choice parsing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
import yaml
from v7_common import EXP,OUT,V6,atomic_json,fingerprint,sha
from primary_ml_cka.data.preprocessing import ensure_canvas


def parse_class_code(raw,class_count):
    if class_count<2:raise ValueError('At least two classes required')
    line=next((s.strip() for s in raw.splitlines() if s.strip()),None)
    if line is None:return dict(label=None,status='empty')
    codes={str(i):i+1 for i in range(class_count)}
    return dict(label=codes.get(line),status='ok' if line in codes else 'invalid')


def classification_prompt(classes):
    options='; '.join(f"{i} {c['name']}" for i,c in enumerate(classes))
    return ('Classify the main object in the image into exactly one of the following '
            f'categories:\n{options}.\nReturn only one integer from 0 to {len(classes)-1}.\n'
            'Do not output reasoning, punctuation, or additional words.')


def validate_dictionary(data,root,check_files=True):
    classes=data['classes'];C=len(classes)
    if data['class_count']!=C:raise ValueError('Declared class count differs from dictionary')
    if C<2 or len({c['id'] for c in classes})!=C:raise ValueError('Distinct classes required')
    if [c['label'] for c in classes]!=list(range(1,C+1)):raise ValueError('Class order/labels must be contiguous')
    if set(data.get('excluded_original_class_ids',[]))&{c['id'] for c in classes}:raise ValueError('Original class entered held-out dictionary')
    reference_ids=set();reference_hashes=set();all_refs=[]
    for c in classes:
        if len(c['references'])!=48:raise ValueError('Exactly48 references/class required')
        for r in c['references']:
            if r['image_id'] in reference_ids or r['sha256'] in reference_hashes:raise ValueError('Duplicate reference image')
            reference_ids.add(r['image_id']);reference_hashes.add(r['sha256']);all_refs.append(r)
    if len({t['id'] for t in data['transitions']})!=len(data['transitions']):raise ValueError('Duplicate transition ID')
    if len(data['transitions'])!=5:raise ValueError('Five frozen directed transitions required')
    for t in data['transitions']:
        if not 1<=t['source_label']<=C or not 1<=t['target_label']<=C or t['source_label']==t['target_label']:raise ValueError('Invalid transition labels')
        if len(t['images'])!=30 or len({r['image_id'] for r in t['images']})!=30:raise ValueError('Exactly30 distinct attack images/transition required')
        for r in t['images']:
            if r['class_id']!=classes[t['source_label']-1]['id']:raise ValueError('Wrong ground-truth source class')
            if r['image_id'] in reference_ids or r['sha256'] in reference_hashes:raise ValueError('Reference/attack overlap')
    if check_files:
        base=Path(root).resolve()
        if 'data_lock_sha256' in data and sha(base/'data_lock.json')!=data['data_lock_sha256']:
            raise ValueError('Changed category data lock')
        for r in all_refs+[r for t in data['transitions'] for r in t['images']]:
            file=(base/r['path']).resolve()
            if not file.is_relative_to(base):raise ValueError('Image path escapes frozen study root')
            if not file.is_file() or sha(file)!=r['sha256']:raise ValueError('Missing/changed canonical image')
    return data


def freeze_json(path,payload):
    path=Path(path)
    if path.exists() and json.loads(path.read_text())!=payload:raise RuntimeError('Changed frozen class selection/manifests')
    if not path.exists():atomic_json(path,payload)


def validate_nested_dictionaries(dictionaries):
    if not dictionaries:raise ValueError('No category dictionaries')
    ordered=sorted(dictionaries,key=lambda x:len(x['classes']));base=ordered[0]
    if len({len(d['classes']) for d in ordered})!=len(ordered):raise ValueError('Duplicate dictionary sizes')
    for data in ordered:
        if data['classes'][:len(base['classes'])]!=base['classes'] or data['transitions']!=base['transitions']:
            raise ValueError('Dictionary scaling changed images, reference IDs or class order')
    return ordered


def prepare(dataset_root,catalog_path,output,counts=(10,20,50),seed=42):
    """ImageFolder input; copy canonical224px PNGs to persistent study storage."""
    dataset_root=Path(dataset_root).resolve();output=Path(output).resolve()
    catalog=json.loads(Path(catalog_path).read_text())
    if not isinstance(catalog,list):raise ValueError('Catalog must be [{id,name},...]')
    if len({c['id'] for c in catalog})!=len(catalog):raise ValueError('Duplicate catalog classes')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',c['id']) for c in catalog):raise ValueError('Unsafe class folder identifier')
    if any(not c.get('name') or '\n' in c['name'] for c in catalog):raise ValueError('One-line class names required')
    if not counts or min(counts)<10 or len(set(counts))!=len(counts):raise ValueError('Distinct dictionary sizes>=10 required')
    excluded=[c['wnid'] for c in yaml.safe_load((V6/'config/runner_template.yaml').read_text())['classes']]
    eligible=[];files={}
    for c in catalog:
        if c['id'] in excluded:continue
        folder=dataset_root/c['id']
        images=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')) if folder.exists() else []
        if len(images)>=78:
            eligible.append(c);files[c['id']]=images
    eligible.sort(key=lambda c:fingerprint(['heldout-classes',seed,c['id']]))
    if len(eligible)<max(counts):raise RuntimeError(f'Dataset blocker: need{max(counts)} non-original classes with>=78 images; found{len(eligible)}')
    chosen=eligible[:max(counts)]
    plan=dict(schema=1,seed=seed,dataset_root=str(dataset_root),catalog_sha256=sha(catalog_path),class_ids=[c['id'] for c in chosen],counts=sorted(counts),references=48,attack_images=30,selection='GT folders and SHA256 ranking only; no model predictions')
    freeze_json(output/'data_lock.json',plan)
    banks=[];attacks={};used_hashes=set()
    for c in chosen:
        candidates=sorted(files[c['id']],key=lambda f:fingerprint(['heldout-images',seed,c['id'],str(f.relative_to(dataset_root))]))[:78]
        rows=[]
        for file in candidates:
            iid=str(file.relative_to(dataset_root));dest=output/'canonical_images'/(fingerprint(iid)+'.png')
            with Image.open(file) as im:
                pixels=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()[None]/255
            canonical=ensure_canvas(pixels,224)[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            digest=hashlib.sha256(canonical.tobytes()).hexdigest()
            if digest in used_hashes:raise ValueError('Duplicate canonical pixels in new-class data')
            used_hashes.add(digest)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if not np.array_equal(np.asarray(Image.open(dest)),canonical):raise RuntimeError('Changed frozen canonical pixels')
            else:Image.fromarray(canonical).save(dest)
            rows.append(dict(image_id=iid,class_id=c['id'],path=str(dest.relative_to(output)),sha256=sha(dest),pixel_sha256=digest))
        banks.append(dict(id=c['id'],name=c['name'],label=len(banks)+1,references=rows[:48]));attacks[c['id']]=rows[48:]
    for count in sorted(counts):
        transitions=[dict(id=f'G{i//2+1:02d}',source_label=i+1,target_label=i+2,images=attacks[banks[i]['id']]) for i in range(0,10,2)]
        data=dict(schema=1,class_count=count,seed=seed,classes=banks[:count],transitions=transitions,excluded_original_class_ids=excluded,data_lock_sha256=sha(output/'data_lock.json'),dictionary_family='nested_held_out_classes',target_prediction_selection=False)
        validate_dictionary(data,output);freeze_json(output/'manifests'/f'classes_{count}.json',data)
    return [output/'manifests'/f'classes_{c}.json' for c in sorted(counts)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,default=OUT/'category_generalization/data');p.add_argument('--class-counts',nargs='+',type=int,default=[10,20,50]);a=p.parse_args()
    for file in prepare(a.dataset_root,a.catalog,a.output,a.class_counts):print(file)


if __name__=='__main__':main()
