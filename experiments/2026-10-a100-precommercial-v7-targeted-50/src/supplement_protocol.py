"""CPU-only deduplication, raw-evidence checks and live supplement reporting."""
import csv
import datetime
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from v7_common import OUT,EXP,TIDS,records,sha,fingerprint,atomic_json
from family_protocol import condition_name,require_excluded_family
from proxy_selection_tables import matrix,writecsv
from supplement_registry import registry

ROOT=OUT/'multiproxy_8h_supplement'


def candidate_key(target,models):
    # Equal independent gradients are mathematically permutation invariant.
    if len(models)!=len(set(models)):raise ValueError('Repeated checkpoint in ensemble')
    return fingerprint([target,sorted(models),'A3-four-depth-margin-patch035-DI-seed42-50steps'])


def candidates():
    reg=registry();selection=matrix();bykey={candidate_key(r['config']['target'],r['config']['models']):r for r in selection}
    unique={};raw=list(csv.DictReader((EXP/'multiproxy_8h_supplement_matrix.csv').open()))
    for r in raw:
        names=r['models'].split(' + ');target=r['target']
        require_excluded_family(names,reg[target]['family'],reg)
        if len(names)!=int(r['proxy_count']):raise ValueError('Candidate proxy count mismatch')
        key=candidate_key(target,names)
        if key not in unique:
            reused=bykey.get(key)
            cfg=reused['config'] if reused else dict(id='XS_'+r['id'],target=target,models=names,method='A3',options={'patch_weight':.35},di=True,fusion='equal',steps=50,seed=42,restarts=1,kind='multiple_proxy_supplement')
            unique[key]=dict(key=key,config=cfg,aliases=[],buckets=[],priority=r['priority'],optional=r['bucket']=='vision_family_extension',source=reused['source'] if reused else 'new_supplement',measurement_condition=reused['measurement_condition'] if reused else condition_name(cfg),source_csv=reused['source_csv'] if reused else str((ROOT/'results/target_direction_results.csv').relative_to(OUT)))
        u=unique[key];u['aliases'].append(r['id']);u['buckets'].append(r['bucket']);u['priority']=min(u['priority'],r['priority'])
    if len(raw)!=52 or len(unique)!=47:raise ValueError('Expected52 display rows and47 unique conditions')
    return list(unique.values())


def evidence(item,deep=False):
    """No target load. Verify exact configs, pinned identity, cohorts and raw hits."""
    from run_v7_attack import METHODS
    reg=registry();c=item['config'];name=item['measurement_condition'];root=OUT/'attacks/test'/name
    if item['source']=='corrected_family_disjoint':lockpath=OUT/'family_disjoint_recovery/method_lock.json'
    elif item['source']=='historical_valid_Gemma_cross_family':lockpath=OUT/'ten_hour_research/selected_method_lock.json'
    elif item['source']=='new_subset_measurement':lockpath=OUT/'proxy_selection/method_lock.json'
    else:lockpath=ROOT/'method_lock.json'
    target=c['target'];rows=[];states=[];detailed=[];complete_attack=True;complete_eval=True
    expected_cfg={**METHODS['A3'],**dict(fusion='equal',input_diversity=True,smoothing=False),**c['options']}
    allowed_optional={'excluded_proxy_family'}
    for tid in TIDS:
        expected=records(OUT/'manifests/test'/f'{tid}.jsonl');expected_ids=[r['image_id'] for r in expected]
        ids=[];measured=[]
        for file in sorted((root/tid).glob('batch_*/state.json')):
            d=json.loads(file.read_text());require_excluded_family(d['models'],reg[target]['family'],reg)
            if d['models']!=c['models'] or d['steps']!=50 or d['seed']!=42 or d['restarts']!=1 or d['target_access'] or d['status']!='attack_complete':raise RuntimeError('Incompatible attack identity/budget')
            if {k:v for k,v in d['config'].items() if k not in allowed_optional}!=expected_cfg:raise RuntimeError('Different A3 loss or preprocessing settings')
            if d['source_label']!=int(tid[1:]) or d['target_label']!=int(tid[1:])+1:raise RuntimeError('Changed source/target class')
            if not len(d['image_ids'])==len(d['clean_paths'])==len(d['adv_paths']):raise RuntimeError('Mismatched image/path lengths')
            if deep:
                for rel,h in d['png_hashes'].items():
                    if sha(OUT/rel)!=h:raise RuntimeError('Frozen PNG hash changed')
                for iid,clean,adv in zip(d['image_ids'],d['clean_paths'],d['adv_paths']):
                    match=next((r for r in expected if r['image_id']==iid),None)
                    if match is None:raise RuntimeError('Unexpected attack image')
                    px=np.asarray(Image.open(OUT/clean));apx=np.asarray(Image.open(OUT/adv))
                    canonical=np.asarray(Image.open(OUT/'canonical_images'/match['relative_path']).convert('RGB'))
                    if px.shape!=(224,224,3) or apx.shape!=px.shape or not np.array_equal(px,canonical):raise RuntimeError('Canonical clean pixels changed')
                    if np.abs(px.astype('int16')-apx.astype('int16')).max()>16:raise RuntimeError('PNG epsilon violation')
                if d['peak_reserved_gib']>75:raise RuntimeError('Reserved VRAM exceeded75GiB')
                for history in d['trace']:
                    if len(history)!=50:raise RuntimeError('Incomplete50-step trace')
                    for step in history:
                        if not all(np.isfinite(step[k]).all() for k in ['losses','margins','gradient_l1','cancellation_ratio']):raise RuntimeError('Nonfinite optimization trace')
            ids+=d['image_ids'];states.append(d)
            metric=OUT/'evaluations'/name/target.replace('/','_')/tid/file.parent.name/'metrics.json'
            if not metric.exists():continue
            m=json.loads(metric.read_text())
            if not lockpath.exists():raise RuntimeError('Evaluation missing its method lock')
            locked=json.loads(lockpath.read_text())
            pinned=locked.get('model_revisions',reg)
            if any(pinned.get(n)!=reg[n] for n in d['models']+[target]):raise RuntimeError('Model revision/preprocessing mismatch')
            if m['method_lock_sha256']!=sha(lockpath) or m['identity']!=fingerprint([d['identity'],d['png_hashes'],reg[target],sha(lockpath)]):raise RuntimeError('Stale evaluation provenance')
            if [x['image_id'] for x in m['images']]!=d['image_ids']:raise RuntimeError('Evaluation cohort differs from attack')
            if any(x['target_hit']!=(x['adv_label']==d['target_label']) or x['clean_correct']!=(x['clean_label']==d['source_label']) for x in m['images']):raise RuntimeError('Wrong success numerator')
            measured+=m['images']
            detailed += [dict(condition=name,target=target,transition=tid,**r) for r in m['images']]
        if len(ids)!=len(set(ids)) or not set(ids)<=set(expected_ids):raise RuntimeError('Duplicate/unexpected frozen attack cohort')
        if ids!=expected_ids:complete_attack=False
        mids=[x['image_id'] for x in measured]
        if len(mids)!=len(set(mids)) or not set(mids)<=set(expected_ids):raise RuntimeError('Duplicate/unexpected measured cohort')
        if mids!=expected_ids:complete_eval=False
        n=len(measured);hits=sum(x['target_hit'] for x in measured);cc=sum(x['clean_correct'] for x in measured);hc=sum(x['target_hit'] and x['clean_correct'] for x in measured)
        rows.append(dict(condition=name,target=target,transition=tid,hits=hits,denominator=n,ASR=hits/n if n else None,clean_correct=cc,clean_correct_hits=hc,clean_correct_ASR=hc/cc if cc else None,parser_success=sum(x['adv_parser']=='ok' for x in measured),status='complete' if mids==expected_ids else 'pending_or_partial',legacy_conditional_TASR='N/A'))
    cos=[];cancellation=[];first=[];last=[];calls={}
    for s in states:
        for counts in s['forward_counts']:
            totals=calls.setdefault(counts['model'],dict(visual_calls=0,backward_calls=0,image_forwards=0))
            for k in totals:totals[k]+=counts[k]
        for history in s['trace']:
            first+=np.asarray(history[0]['margins']).mean(1).tolist();last+=np.asarray(history[-1]['margins']).mean(1).tolist()
            for step in history:
                cos+=np.asarray(step['pairwise_cosines']).ravel().tolist();cancellation+=step['cancellation_ratio']
    return dict(attack_complete=complete_attack,evaluation_complete=complete_eval,images=sum(r['denominator'] for r in rows),attack_images=sum(len(s['image_ids']) for s in states),rows=rows,detailed=detailed,attack_seconds=sum(s['runtime_seconds'] for s in states),peak_reserved_gib=max((s['peak_reserved_gib'] for s in states),default=None),mean_gradient_cosine=float(np.mean(cos)) if cos else None,negative_cosine_fraction=float(np.mean(np.asarray(cos)<0)) if cos else None,mean_cancellation=float(np.mean(cancellation)) if cancellation else None,mean_initial_margin=float(np.mean(first)) if first else None,mean_final_margin=float(np.mean(last)) if last else None,forward_backward_counts=calls)


def inventory():
    info={}
    for name,cmd in {'hostname':['hostname'],'date':['date','-u','--iso-8601=seconds'],'gpu':['nvidia-smi'],'processes':['ps','-eo','pid,ppid,user,etime,args'],'tmux':['tmux','list-sessions'],'git':['git','rev-parse','HEAD']}.items():
        result=subprocess.run(cmd,capture_output=True,text=True);value=result.stdout
        if name=='processes':value='\n'.join(s for s in value.splitlines() if any(t in s for t in ['run_family_disjoint','run_v7_attack.py','evaluate_v7.py','run_multiproxy_supplement']))
        info[name]=value
    info['captured_at']=time.time();atomic_json(ROOT/'audits/server_inventory.json',info)
    writecsv(ROOT/'reports/multiproxy_8h_server_inventory.csv',[dict(component=k,observed_at=info['captured_at'],detail=v) for k,v in info.items() if isinstance(v,str)])
    return info


def snapshot(items,owner_active=False,deep=False):
    queue=[];results=[];detailed=[];statepath=ROOT/'state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
    current=json.loads((OUT/'family_disjoint_recovery/current_stage.json').read_text())
    for item in items:
        try:
            e=evidence(item,deep=deep);status='COMPLETE' if e['evaluation_complete'] else 'PARTIAL' if e['images'] or e['attack_images'] else 'QUEUED'
            reason='validated frozen attacks and raw target answers' if status=='COMPLETE' else 'waiting for existing GPU owner' if owner_active else 'not yet measured'
            if item['source']=='corrected_family_disjoint' and owner_active and not e['evaluation_complete']:status='RUNNING';reason='existing correction owns this condition'
            if item['config']['id']==current.get('condition'):reason='currently executing in existing campaign'
            if item['optional'] and status!='COMPLETE':reason='optional; requires BF16/gradient smoke and surrogate-only dev selection'
            if item['config']['id'] in state.get('blocked',{}):status='BLOCKED';reason=state['blocked'][item['config']['id']]
            for row in e['rows']:results.append(dict(candidate_aliases='|'.join(item['aliases']),proxy_names=' + '.join(item['config']['models']),K=len(item['config']['models']),**row))
            detailed+=e['detailed']
            hits=sum(r['hits'] for r in e['rows']);N=e['images']
            q=dict(unique_key=item['key'],aliases='|'.join(item['aliases']),buckets='|'.join(item['buckets']),target=item['config']['target'],models=' + '.join(item['config']['models']),K=len(item['config']['models']),priority=item['priority'],status=status,action=reason,attack_images=e['attack_images'],evaluated_images=N,hits=hits,ASR=hits/150 if status=='COMPLETE' else None,measurement_condition=item['measurement_condition'],source=item['source'],attack_seconds=e['attack_seconds'],peak_reserved_gib=e['peak_reserved_gib'])
        except Exception as exc:q=dict(unique_key=item['key'],aliases='|'.join(item['aliases']),buckets='|'.join(item['buckets']),target=item['config']['target'],models=' + '.join(item['config']['models']),K=len(item['config']['models']),priority=item['priority'],status='BLOCKED',action=str(exc),attack_images=0,evaluated_images=0,hits=None,ASR=None,measurement_condition=item['measurement_condition'],source=item['source'],attack_seconds=None,peak_reserved_gib=None)
        admission_path=ROOT/'audits'/(item['config']['id']+'_admission.json')
        q['estimated_attack_seconds']=json.loads(admission_path.read_text())['estimated_attack_seconds'] if admission_path.exists() else None
        q['GPU_owner']='existing_family_disjoint_campaign' if owner_active else 'supplement_scheduler_or_idle'
        for k in ['mean_gradient_cosine','negative_cosine_fraction','mean_cancellation','mean_initial_margin','mean_final_margin']:
            q[k]=e[k] if q['status']!='BLOCKED' else None
        q['forward_backward_counts']=json.dumps(e['forward_backward_counts'],sort_keys=True) if q['status']!='BLOCKED' else None
        queue.append(q)
    writecsv(ROOT/'reports/multiproxy_8h_queue.csv',queue);writecsv(ROOT/'reports/multiproxy_8h_results.csv',results)
    writecsv(ROOT/'results/per_image_results.csv',detailed)
    for bucket,title in [('original_multiple','original_multiple_completion'),('selection_7subsets','selection_completion'),('vision_family_extension','extension_completion')]:writecsv(ROOT/'reports'/f'{title}.csv',[r for r in queue if bucket in r['buckets'].split('|')])
    budget=json.loads((ROOT/'budget.json').read_text());remaining=max(0,(budget['deadline']-time.time())/3600)
    lines=['# Multiple Proxy supplement: live report','',f'Captured UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}.',f'Budget remaining: {remaining:.2f} hours. GPU owned by existing correction: {owner_active}.','',f'52 display rows deduplicated to {len(items)} target-specific conditions. Complete: {sum(r["status"]=="COMPLETE" for r in queue)}. Missing is not zero.','', 'Existing32-condition correction and loss ablations are preserved. No new historical single/layer campaign is launched. Optional extension selection uses only held-out-surrogate development features, never actual target outcomes.','', '| Aliases | Target | K | Status | Hits/N | ASR |','|---|---|---|---|---|---|']
    for r in queue:
        score=f'{r["ASR"]:.1%}' if r['ASR'] is not None else 'N/A'
        lines.append(f'| {r["aliases"]} | {r["target"]} | {r["K"]} | {r["status"]} | {r["hits"]}/{r["evaluated_images"]} | {score} |')
    lines+=['','Full raw PNGs/caches remain on persistent root disk. Candidate completion does not imply the whole campaign is finished. Commercial pilot remains separately gated by verified model/route/authentication/spend.','', 'This window includes time waiting for the existing single-GPU campaign. It does not promise every candidate will fit. Reuse provenance, exact model and method locks accompany the compact tables.']
    pairs=OUT/'family_disjoint_recovery/results/paired_single_comparisons.csv'
    if pairs.exists():
        lines+=['','## Existing matched paired comparisons','','Ensemble minus member single, stratified paired bootstrap95% intervals. Descriptive comparisons only; no target-guided algorithm selection.','','| Target | Single | ASR gain |95% interval |','|---|---|---|---|']
        paired=list(csv.DictReader(pairs.open()));writecsv(ROOT/'results/existing_paired_single_comparisons.csv',paired)
        for r in paired:lines.append(f'| {r["target"]} | {r["single"]} | {float(r["ASR_difference"]):.1%} | [{float(r["CI_low"]):.1%}, {float(r["CI_high"]):.1%}] |')
    (ROOT/'reports/multiproxy_8h_report.md').write_text('\n'.join(lines)+'\n')
    return queue
