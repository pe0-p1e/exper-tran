"""Evidence-backed setting inventory; frozen V6 artifacts are read-only."""
import csv
import importlib.metadata
import json
import platform
import numpy as np
from pp4_protocol import OUT,EXP,records,sha,atomic_json,fingerprint
from pp4_campaign import table
from supplement_registry import registry
ROOT=OUT/'v6_v7_setting_audit'
OLD=OUT.parent/'a100_precommercial_v6_multi_repr_recovery'


def main():
    checks={};images=[]
    for tid in ['T01','T03','T05','T07','T09']:
        a=OLD/'manifests/attack_cohorts'/f'{tid}.jsonl';b=OUT/'manifests/test'/f'{tid}.jsonl'
        checks['test_'+tid]=dict(V6_sha256=sha(a),V7_sha256=sha(b),identical=sha(a)==sha(b))
        images+=records(a)
    for i in range(1,11):
        a=OLD/'manifests/class_references'/f'class_references_{i:02d}.jsonl';b=OUT/'manifests/class_references'/a.name
        checks['reference_'+str(i)]=dict(V6_sha256=sha(a),V7_sha256=sha(b),identical=sha(a)==sha(b))
        images+=records(a)
    different=[r['image_id'] for r in images if sha(OLD/'canonical_images'/r['relative_path'])!=sha(OUT/'canonical_images'/r['relative_path'])]
    checks['canonical_images']=dict(checked=len(images),different_ids=different)
    if different or any(not v['identical'] for k,v in checks.items() if 'identical' in v):raise RuntimeError('Input identity mismatch')
    oldreg=json.loads((EXP.parent/'2026-10-a100-precommercial-v6/model_revisions.json').read_text());newreg=registry()
    checks['model_revisions']=dict(checked=len(oldreg),changed=[n for n in oldreg if oldreg[n]!=newreg[n]])
    oldrows={}
    for path in (OLD/'multiple_proxy/evaluations').glob('*/T*.json'):
        d=json.loads(path.read_text())
        for r in d['images']:oldrows[(d['ensemble_id'].split('_')[0],d['target'],d['transition_id'],r['image_id'])]=r
    newrows=list(csv.DictReader((OUT/'pull_push_original13/reports/per_image.csv').open()))
    pairs=[(r,oldrows.get((r['ensemble'],r['target'],r['transition'],r['image_id']))) for r in newrows]
    checks['clean_target_predictions']=dict(new_rows=len(newrows),matched=sum(o is not None for _,o in pairs),
        label_mismatches=sum((None if r['clean_label']=='' else int(r['clean_label']))!=o['clean_label'] for r,o in pairs if o is not None),
        raw_response_mismatches=sum(r['clean_raw']!=o['clean_raw'] for r,o in pairs if o is not None),
        missing_labels=sum(r['clean_label']=='' for r,_ in pairs),
        note='Clean labels only; changed adversarial predictions are expected and not compared for equality')
    oldaudits=json.loads((OLD/'audits/model_dtype_audit.json').read_text())
    checks['recovery_BF16']=dict(load_events=len(oldaudits),failed=sum(a['status']!='pass' for a in oldaudits),
        floating_parameter_dtypes=sorted({k for a in oldaudits for k in a['parameter_counts_by_dtype']}))
    checks['current_environment']=dict(python=platform.python_version(),**{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
        historical_environment_full_replay='NOT_ESTABLISHED; current same-source parity is not historical binary replay')
    cache_rows=[]
    for entry in json.loads((ROOT/'reports/reference_cache_parity.json').read_text()):
        spec=newreg[entry['model']];depth=spec['expected_vision_depth']
        from vision_features import layer_indices
        layers=layer_indices(depth,[.25,.5,.75,1.])
        key=fingerprint(dict(model=spec,refs=[sha(OUT/'manifests/class_references'/f'class_references_{i:02d}.jsonl') for i in range(1,11)],layers=layers,k=8,version='v7_features_2_v6_pooling'))
        path=OUT/'cache/reference_features'/f'{key}.npz'
        with np.load(entry['historical_cache']) as z:a=np.stack([z[f'class_{i}'] for i in range(10)]).astype(np.float64)
        with np.load(path) as z:b=z[f'z_{layers[-1]}'].astype(np.float64)
        def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
        ac=unit(unit(a).mean(1));bc=unit(unit(b).mean(1))
        for i in [0,2,4,6,8]:
            ad=ac[i+1]-ac[i];bd=bc[i+1]-bc[i]
            cache_rows.append(dict(model=entry['model'],transition='T'+str(i+1).zfill(2),
                old_center_gap_norm=float(np.linalg.norm(ad)),new_center_gap_norm=float(np.linalg.norm(bd)),
                center_difference_direction_cosine=float(np.dot(unit(ad),unit(bd))),
                V7_cache=str(path),V7_cache_sha256=sha(path),
                note='Near-identical reference vectors do not alone prove equivalent class-difference directions'))
    table(ROOT/'reports/prototype_direction_parity.csv',cache_rows)
    atomic_json(ROOT/'reports/identity_checks.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
