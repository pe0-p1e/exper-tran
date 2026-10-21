#!/usr/bin/env python3
"""Reproduce a measured conditional model-layer path; CPU/read-only inputs."""
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from selection import score
IDS=['V8_1302a3c70a920d9a9c','V8_0804406e1d56ce93f9','V8_b995b713d868291b86','V8_e4d755260a585ffcf4','V8_04b95758d68a6db9a8']
def generate():
    rows=[];previous=None
    for cid in IDS:
        c=C.read(C.ROOT/'conditions'/(cid+'.json'))
        if not c or c['family']!='gemma4' or c['split']!='screen' or c['validator']!='Gemma 4 E4B-it':raise ValueError('Wrong condition')
        C.legal(c['layers'],c['validator'])
        if not C.attack_done(c) or not C.eval_done(c):raise ValueError('Incomplete/corrupted evidence '+cid)
        v=C.read(C.eval_path(c));images={x['image_id']:x for x in v['images']};paired=None
        if previous is not None:
            if previous.keys()!=images.keys():raise ValueError('Cohorts differ')
            for i,x in images.items():
                if any(x[k]!=previous[i][k] for k in ['source','target','transition','clean_label']):raise ValueError('Labels differ')
            paired=dict(gained=sum(x['target_hit'] and not previous[i]['target_hit'] for i,x in images.items()),lost=sum(not x['target_hit'] and previous[i]['target_hit'] for i,x in images.items()))
        rows.append(dict(condition=cid,layers=c['layers'],proxy_count=len(c['layers']),tap_count=sum(map(len,c['layers'].values())),
           hits=v['hits'],N=v['N'],directions=score(c)['directions'],score=score(c),paired_with_previous=paired,
           evidence_sha256=C.sha(C.eval_path(c))))
        previous=images
    result=dict(validator='Gemma 4 E4B-it',development_only=True,used_for_selection=True,rows=rows,
        conclusion='Three units match twelve-unit uniform9/10; fourth unit adds no success. Not a confirmed ASR improvement or proof of lower walltime.',
        actual_gradient_fusion='mean layer losses within each proxy, then per-image L1 gradient normalization, then equal proxy mean')
    C.atomic(C.ROOT/'reports/joint_selection_path.json',result)
    return result
if __name__=='__main__':print(json.dumps(generate(),indent=2))
