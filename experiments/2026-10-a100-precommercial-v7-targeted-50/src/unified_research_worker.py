"""One admitted BF16 GPU job: development attack, validator, or embedding cache."""
import argparse
import json
import sys
from pathlib import Path

import proxy_study_worker  # Existing memory cap, isolated BF16 audit and reference writer locks.
from proxy_study_protocol import OUT, ROOT, sha, atomic_json, fingerprint
from unified_research_plan import conditions, condition_name, RESEARCH


def filter_dev(module):
    original=module.records
    def selected(path):
        rows=original(path)
        return rows[2:4] if Path(path).parent==OUT/'manifests/dev' else rows
    module.records=selected


def embeddings(c,name):
    import numpy as np
    import torch
    from vision_features import VisionProxy
    from primary_ml_cka.artifacts.png import load_png_tensor
    dest=RESEARCH/'embeddings'/c['id'];meta=dest/(name.replace('/','_')+'.json')
    if meta.exists():
        old=json.loads(meta.read_text())
        if sha(Path(old['npz']))==old['npz_sha256']:return
        raise RuntimeError('Changed embedding cache')
    root=OUT/'attacks/dev'/condition_name(c)
    states=[json.loads(p.read_text()) for p in sorted(root.glob('T*/batch_*/state.json'))]
    count=sum(len(d['image_ids']) for d in states)
    if count!=10:raise RuntimeError('Embedding extraction requires all10 frozen images')
    model=VisionProxy(name,[.25,.5,.75,1.],role='surrogate_followup_posthoc');model.prepare_bank()
    payload={};ids=[];tids=[];source=[];target=[];input_hashes={}
    try:
        for l in model.layers:
            payload[f'clean_{l}']=[];payload[f'adv_{l}']=[];payload[f'refs_{l}']=model.bank[l][0].float().cpu().numpy()
        for d in states:
            ids.extend(d['image_ids']);tids.extend([d['transition_id']]*len(d['image_ids']))
            source.extend([d['source_label']]*len(d['image_ids']));target.extend([d['target_label']]*len(d['image_ids']))
            for kind,key in [('clean','clean_paths'),('adv','adv_paths')]:
                for rel in d[key]:
                    if sha(OUT/rel)!=d['png_hashes'][rel]:raise RuntimeError('Changed attack input')
                    input_hashes[rel]=sha(OUT/rel)
                with torch.inference_mode():
                    fs=model.features(torch.stack([load_png_tensor(OUT/p) for p in d[key]]).cuda())
                for l in model.layers:payload[f'{kind}_{l}'].append(fs[l][0].float().cpu().numpy())
        for l in model.layers:
            for kind in ['clean','adv']:payload[f'{kind}_{l}']=np.concatenate(payload[f'{kind}_{l}'])
        payload.update(image_ids=np.array(ids),transitions=np.array(tids),source_labels=np.array(source),target_labels=np.array(target))
        dest.mkdir(parents=True,exist_ok=True);npz=meta.with_suffix('.npz');tmp=npz.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**payload);tmp.replace(npz)
        atomic_json(meta,dict(model=name,revision=model.spec['revision'],condition=c['id'],N=10,layers=model.layers,
             npz=str(npz),npz_sha256=sha(npz),input_hashes=input_hashes,posthoc_only=True,target_feedback_for_selection=False))
    finally:model.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['attack','validate','embeddings'])
    p.add_argument('--condition',required=True);p.add_argument('--model');a=p.parse_args()
    c=next(c for c in conditions() if c['id']==a.condition)
    if a.mode=='attack':
        import run_v7_attack as runner
        filter_dev(runner);runner.prepare_data=proxy_study_worker.verify_existing_data
        sys.argv=[sys.argv[0],'--models',*c['models'],'--method','A3','--fusion',c['fusion'],
                  '--split','dev','--steps','50','--seed','42','--batch-size','2','--restarts','1',
                  '--input-diversity','--limit-per-direction','2','--run-label',c['id'],
                  '--config-json',str(RESEARCH/'configs'/(c['id']+'.json'))]
        runner.main()
    elif a.mode=='validate':
        import evaluate_surrogate_outputs as runner
        filter_dev(runner)
        sys.argv=[sys.argv[0],'--root',str(OUT/'attacks/dev'/condition_name(c)),'--models',a.model,'--limit-per-direction','2']
        runner.main()
    else:embeddings(c,a.model)


if __name__=='__main__':main()
