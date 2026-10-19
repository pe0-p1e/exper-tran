#!/usr/bin/env python3
"""Export three GT-selected cached pairs without GPU or commercial inference."""
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image
import yaml
from v7_common import OUT,EXP,V6,records,sha,fingerprint,atomic_json
from common import classification_prompt


def main():
    root=OUT/'commercial_pilot3';root.mkdir(parents=True,exist_ok=True)
    old=OUT/'ten_hour_research/selected_method_lock.json'
    choice=json.loads(old.read_text());attack=Path(choice['selected']['root'])
    prompt=classification_prompt(yaml.safe_load((V6/'config/runner_template.yaml').read_text()))
    cases=[];requests=[]
    for tid in ['T03','T05','T09']:
        chosen=records(OUT/'manifests/dev'/f'{tid}.jsonl')[0]['image_id']
        found=[]
        for path in (attack/tid).glob('batch_*/state.json'):
            d=json.loads(path.read_text())
            if chosen in d['image_ids']:found.append((path,d,d['image_ids'].index(chosen)))
        if len(found)!=1:raise RuntimeError('Missing or duplicated frozen pilot image '+chosen)
        path,d,i=found[0]
        pair={}
        arrays=[]
        for kind,key in [('clean','clean_paths'),('adversarial','adv_paths')]:
            src=OUT/d[key][i]
            if sha(src)!=d['png_hashes'][str(src.relative_to(OUT))]:raise RuntimeError('Frozen PNG changed')
            with Image.open(src) as image:
                if image.info:raise RuntimeError('PNG metadata requires separate provenance review before blinding')
                arrays.append(np.asarray(image).astype('int16'))
            rid=fingerprint(['commercial-connectivity-pilot-v1',tid,kind,chosen])[:16]
            dest=root/'blinded'/rid;dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest/'image.png');(dest/'prompt.txt').write_text(prompt+'\n')
            request=dict(request_id=rid,image_sha256=sha(dest/'image.png'),prompt_sha256=sha(dest/'prompt.txt'))
            requests.append(request);pair[kind]=rid
        max_delta=int(np.abs(arrays[0]-arrays[1]).max())
        if max_delta>16:raise RuntimeError('L-infinity violation')
        cases.append(dict(transition=tid,image_id=chosen,source_label=d['source_label'],target_label=d['target_label'],requests=pair,max_uint8_delta=max_delta,source_state=str(path.relative_to(OUT)),source_models=d['models'],steps=d['steps'],seed=d['seed']))
    manifest=dict(status='prepared_no_commercial_measurements',N_unique_images=3,N_requests_per_model=6,selection='first fixed GT-only dev image in T03/T05/T09; no output filtering',method_lock_sha256=sha(old),cases=cases,requests=sorted(requests,key=lambda x:x['request_id']),task='fixed_10_class_recognition',not_for_algorithm_selection=True)
    lock=root/'private/manifest.json'
    if lock.exists() and json.loads(lock.read_text())!=manifest:raise RuntimeError('Never overwrite a changed pilot cohort')
    atomic_json(lock,manifest)
    atomic_json(root/'request_order.json',[x['request_id'] for x in manifest['requests']])
    archive=root/'pilot3_blinded_inputs.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted((root/'blinded').rglob('*')):
            if file.is_file():z.write(file,file.relative_to(root/'blinded'))
    (root/'README.md').write_text('''# Three-image commercial connectivity pilot

Prepared six separate image requests (three clean/adversarial pairs). No commercial
results have been measured. Files have opaque request IDs; target/source labels
are only in private/manifest.json and must never be supplied to a target client.

For every image use a fresh isolated session, the exact prompt.txt and only image.png.
Do not expose this repository, other requests, private manifests or attack status.
Record actual model ID, CLI version, route, UTC time, raw response and evidence that
one image was attached. Do not reuse conversation history or switch models on retry.

pilot3_blinded_inputs.zip contains only image.png/prompt.txt under opaque folders.
This is a limited three-direction interface smoke, not a completed scientific
ablation or a statistical transferability claim. The GPU campaign runs separately.
''')
    atomic_json(root/'preparation.json',dict(status='ready_for_authenticated_isolated_target_clients',archive_sha256=sha(archive),images=3,requests_per_model=6,commercial_inference_calls=0))
    print('Prepared',archive,'(3 images, 6 isolated requests/model; no inference)')


if __name__=='__main__':main()
