"""V7 paths, immutable data preparation, and atomic artifact helpers."""
from pathlib import Path
import hashlib
import json
import os
import sys

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
V6 = REPO / 'experiments/2026-10-a100-precommercial-v6'
BASE = REPO / 'outputs/a100_precommercial_v6_multi_repr_recovery'
OUT = REPO / 'outputs/a100_precommercial_v7_targeted50'
sys.path[:0] = [str(REPO/'src'), str(V6/'src'), str(REPO/'experiments/2026-09-a100-final-v5/src')]
REGISTRY = json.loads((V6/'model_revisions.json').read_text())
TIDS = ['T01','T03','T05','T07','T09']
TARGETS = ['Qwen3.5-27B','InternVL3.5-14B-HF','Gemma 4 31B-it']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_data():
    from prepare_recovery_data import atomic_copy
    ids = set()
    for label in range(1,11):
        src = BASE/'manifests/class_references'/f'class_references_{label:02d}.jsonl'
        atomic_copy(src, OUT/'manifests/class_references'/src.name)
        bank = records(src)
        assert len(bank)==48
        ids.update(r['image_id'] for r in bank)
        for r in bank:
            atomic_copy(BASE/'canonical_images'/r['relative_path'], OUT/'canonical_images'/r['relative_path'])
    test_ids = set()
    dev_ids = set()
    for tid in TIDS:
        src=BASE/'manifests/attack_cohorts'/f'{tid}.jsonl'
        atomic_copy(src, OUT/'manifests/test'/src.name)
        test=records(src)
        assert len(test)==30
        assert not ids.intersection(r['image_id'] for r in test)
        test_ids.update(r['image_id'] for r in test)
        for r in test:
            atomic_copy(BASE/'canonical_images'/r['relative_path'], OUT/'canonical_images'/r['relative_path'])
    for tid in TIDS:
        candidates=records(REPO/'outputs/a100_precommercial_v6/evaluation/manifests/transitions'/tid/'candidates.jsonl')
        pool=[r for r in candidates if r['image_id'] not in ids|test_ids]
        chosen=sorted(pool,key=lambda r:fingerprint(['v7dev',42,tid,r['image_id']]))[:10]
        assert len(chosen)==10, (tid,len(chosen))
        chosen.sort(key=lambda r:r['image_id'])
        dst=OUT/'manifests/dev'/f'{tid}.jsonl'; dst.parent.mkdir(parents=True,exist_ok=True)
        payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in chosen)
        if dst.exists() and dst.read_text()!=payload: raise RuntimeError('Changed dev manifest')
        dst.write_text(payload)
        dev_ids.update(r['image_id'] for r in chosen)
        for r in chosen:
            atomic_copy(REPO/'outputs/a100_precommercial_v6/canonical_images'/r['relative_path'], OUT/'canonical_images'/r['relative_path'])
    assert not dev_ids & (test_ids|ids)
    atomic_json(OUT/'audits/data_split.json',dict(test_count=len(test_ids),dev_count=len(dev_ids),reference_count=len(ids),overlap=False,selection='GT only; SHA256 deterministic; no model predictions'))
    atomic_json(OUT/'audits/model_revisions.json', REGISTRY)


def images(batch, device='cuda'):
    import numpy as np
    import torch
    from PIL import Image
    from primary_ml_cka.data.preprocessing import ensure_canvas
    arr=[]
    for r in batch:
        with Image.open(OUT/'canonical_images'/r['relative_path']) as im:
            value=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float().to(device)/255
        arr.append(ensure_canvas(value.unsqueeze(0),224).squeeze(0))
    return torch.stack(arr).mul(255).round().div(255)
