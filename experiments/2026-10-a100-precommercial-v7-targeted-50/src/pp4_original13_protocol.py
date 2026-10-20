"""Original workbook's13 rows, frozen four-layer PP.5 test campaign."""
import csv
import json
from pathlib import Path
from pp4_protocol import OUT,PREVIOUS,EXP,condition,sha,fingerprint,atomic_json,records,TIDS
from supplement_registry import registry

ROOT=OUT/'pull_push_original13'


def plan():
    from supplement_protocol import candidates
    from family_protocol import condition_name,require_excluded_family
    old={alias:item for item in candidates() for alias in item['aliases']}
    raw=[r for r in csv.DictReader((EXP/'multiproxy_8h_supplement_matrix.csv').open()) if r['bucket']=='original_multiple']
    result=[]
    for row in raw:
        item=old[row['id']];models=item['config']['models'];target=row['target']
        if set(models)!=set(row['models'].split(' + ')):raise RuntimeError('Original composition differs')
        require_excluded_family(models,registry()[target]['family'],registry())
        c=condition(models,pull=.5)
        c['options'].pop('pp4_dev_offset')
        c['options'].update(excluded_proxy_family=registry()[target]['family'],
            pp4_full_worker_sha256=sha(EXP/'src/pp4_original13_worker.py'))
        c.update(id='PP4R_'+row['id']+'_'+fingerprint([models,c['options']])[:12],split='test',target=target)
        c['name']=condition_name(c)
        result.append(dict(id=row['id'],target=target,condition=c,previous_A3=item))
    if len(result)!=13 or len({(r['target'],tuple(r['condition']['models'])) for r in result})!=13:
        raise RuntimeError('Expected13 unique workbook rows')
    return result


def smoke():
    c=condition(['InternVL3.5-2B-HF'],pull=.5,steps=2)
    c['options']['pp4_full_worker_sha256']=sha(EXP/'src/pp4_original13_worker.py')
    c['id']='PP4R_smoke_'+fingerprint(c['options'])[:12]
    from family_protocol import condition_name
    c['name']=condition_name(c)
    return c


def freeze():
    from commercial_matrix_pack import freeze as immutable
    from primary_ml_cka.models.common.loading import local_snapshot,safetensor_files
    ROOT.mkdir(parents=True,exist_ok=True);rows=plan();reg=registry()
    sources=['pp4_original13_protocol.py','pp4_original13_worker.py','pp4_protocol.py','pp4_objective.py','pp4_worker.py',
        'run_v7_attack.py','vision_features.py','representation_objectives.py','attack_optimizers.py','evaluate_v7.py']
    names={n for r in rows for n in r['condition']['models']}|{r['target'] for r in rows}
    cache=Path('/home/ubuntu/exper-tran/.hf-cache')
    snapshots={}
    for name in sorted(names):
        spec=reg[name];path=local_snapshot(cache,spec['repo_id'],spec['revision'])
        shards=safetensor_files(path)
        if any(not s.is_file() or s.stat().st_size==0 for s in shards):raise RuntimeError('Missing checkpoint shard')
        for index in path.glob('*.safetensors.index.json'):
            expected=set(json.loads(index.read_text())['weight_map'].values())
            if any(not (path/name).is_file() for name in expected):raise RuntimeError('Missing indexed checkpoint shard')
        snapshots[name]=dict(revision=spec['revision'],snapshot=str(path),shards=len(shards))
    refs={r['image_id'] for p in (OUT/'manifests/class_references').glob('*.jsonl') for r in records(p)}
    if len(refs)!=480:raise RuntimeError('Reference count differs')
    ids={t:[r['image_id'] for r in records(OUT/'manifests/test'/f'{t}.jsonl')] for t in TIDS}
    if any(len(v)!=30 or len(set(v))!=30 or set(v)&refs for v in ids.values()):raise RuntimeError('Wrong test cohort')
    value=dict(protocol='pp4_original_workbook13_v1',rows=rows,smoke=smoke(),test_ids=ids,
        test_manifest_sha256={t:sha(OUT/'manifests/test'/f'{t}.jsonl') for t in TIDS},
        reference_manifest_sha256={p.name:sha(p) for p in sorted((OUT/'manifests/class_references').glob('*.jsonl'))},
        model_revisions=reg,snapshots=snapshots,source_hashes={n:sha(EXP/'src'/n) for n in sources},
        model_parameter_dtype='bfloat16',quantization=False,pull=.5,push=.5,patch=0,CLS=0,
        layers=[.25,.5,.75,1.],DI=True,seed=42,steps=50,epsilon='16/255',step_size='1/255',momentum=1,
        random_start=True,images_per_direction=30,directions=5,expected_rows=1950,
        tuning=False,target_feedback_for_selection=False,commercial_testing=False,
        comparison='Method-level comparison vs four-layer Margin+Patch+DI and historical deepest PP/noDI; not weight-only ablation')
    immutable(ROOT/'protocol_lock.json',value)
    for c in [r['condition'] for r in rows]+[smoke()]:atomic_json(ROOT/'configs'/(c['id']+'.json'),c['options'])
    for target in {r['target'] for r in rows}:
        group=[r for r in rows if r['target']==target]
        immutable(ROOT/'target_locks'/(target.replace('/','_')+'.json'),dict(protocol='strict_family_disjoint',model_revisions=reg,
            allowed_test_conditions=[r['condition']['name'] for r in group],
            condition_targets={r['condition']['name']:target for r in group},
            condition_models={r['condition']['name']:r['condition']['models'] for r in group},
            pp4_protocol_sha256=sha(ROOT/'protocol_lock.json')))
    return rows


def evidence(c,deep=False):
    from pp4_protocol import development_ids
    ids=development_ids() if c['split']=='dev' else json.loads((ROOT/'protocol_lock.json').read_text())['test_ids']
    if c['steps']!=50:ids={'T01':ids['T01']}
    found=set();measured=[];seconds=0.;peak=0.
    lock=ROOT/'target_locks'/(c.get('target','').replace('/','_')+'.json')
    for path in sorted((OUT/'attacks'/c['split']/c['name']).glob('T*/batch_*/state.json')):
        d=json.loads(path.read_text());tid=d['transition_id']
        expected={**c['options'],'fusion':'equal','input_diversity':True,'smoothing':False}
        if d['config']!=expected or d['models']!=c['models'] or d['steps']!=c['steps'] or d['seed']!=42 or d['target_access']:
            raise RuntimeError('Attack protocol mismatch')
        if d['linf_png']>16/255+1e-7:raise RuntimeError('PNG bound violation')
        for iid in d['image_ids']:
            if (tid,iid) in found or tid not in ids or iid not in ids[tid]:raise RuntimeError('Wrong/duplicate image')
            found.add((tid,iid))
        if deep:
            for rel,h in d['png_hashes'].items():
                if sha(OUT/rel)!=h:raise RuntimeError('Changed PNG')
        seconds+=d['runtime_seconds'];peak=max(peak,d['peak_reserved_gib'])
        if c['split']!='test':continue
        metrics=OUT/'evaluations'/c['name']/c['target'].replace('/','_')/tid/path.parent.name/'metrics.json'
        if not metrics.exists():continue
        m=json.loads(metrics.read_text())
        if m['identity']!=fingerprint([d['identity'],d['png_hashes'],registry()[c['target']],sha(lock)]):raise RuntimeError('Stale target evaluation')
        if [r['image_id'] for r in m['images']]!=d['image_ids']:raise RuntimeError('Evaluation cohort differs')
        for r in m['images']:
            if r['target_hit']!=(r['adv_label']==d['target_label']):raise RuntimeError('Wrong success numerator')
            measured.append(dict(transition=tid,**r))
    expected={(t,i) for t,v in ids.items() for i in v}
    return dict(attack_complete=found==expected,attack_images=len(found),evaluation_complete=len(measured)==len(expected),
        measured=measured,attack_seconds=seconds,peak_reserved_gib=peak)
