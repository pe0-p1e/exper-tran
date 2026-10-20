"""One GPU owner for the bounded paired composition study; no commercial clients."""
import json
import time
import numpy as np
import pp4_campaign as scheduler
import pp4_composition_plan as protocol


def report(p,state):
    rows=[];images=[];observed={}
    for row in p['ladder']:
        c=row['condition'];a,v=scheduler.evidence(c);done=a['complete'] and len(v)==len(protocol.VALIDATORS)
        hits=sum(r['target_hit'] for d in v.values() for r in d['images'])
        rows.append(dict(composition=row['group'],condition=c['id'],models=' + '.join(c['models']),K=len(c['models']),
            pull=c['options']['pull_weight'],push=c['options']['push_weight'],status='COMPLETE' if done else 'PARTIAL' if a['N'] else 'QUEUED',
            attack_images=a['N'],validated_model_images=sum(d['denominator'] for d in v.values()),
            hits=hits if done else None,denominator=30,ASR=hits/30 if done else None,reused=bool(a.get('reused')),
            attack_seconds=a['seconds']))
        if done:
            observed[(row['group'],c['options']['pull_weight'])]={(m,r['image_id']):r for m,d in v.items() for r in d['images']}
        for m,d in v.items():
            images.extend(dict(composition=row['group'],pull=c['options']['pull_weight'],validator=m,**r) for r in d['images'])
    paired=[]
    for name,_ in protocol.COMBINATIONS:
        if (name,.5) not in observed or (name,.75) not in observed:continue
        left,right=observed[(name,.75)],observed[(name,.5)]
        if left.keys()!=right.keys():raise RuntimeError('Paired cohort differs')
        ids=sorted({k[1] for k in left});delta=np.array([np.mean([int(right[k]['target_hit'])-int(left[k]['target_hit']) for k in left if k[1]==iid]) for iid in ids])
        boot=np.random.default_rng(42).choice(delta,size=(4000,len(ids)),replace=True).mean(1)
        paired.append(dict(composition=name,N_unique_images=len(ids),model_image_pairs=len(left),
            hits_050=sum(r['target_hit'] for r in right.values()),hits_075=sum(r['target_hit'] for r in left.values()),
            paired_wins=sum(right[k]['target_hit'] and not left[k]['target_hit'] for k in left),
            paired_losses=sum(left[k]['target_hit'] and not right[k]['target_hit'] for k in left),
            delta_ASR=float(delta.mean()),image_cluster_bootstrap_low=float(np.quantile(boot,.025)),image_cluster_bootstrap_high=float(np.quantile(boot,.975))))
    scheduler.table(protocol.ROOT/'reports/conditions.csv',rows);scheduler.table(protocol.ROOT/'reports/per_image.csv',images)
    scheduler.table(protocol.ROOT/'reports/paired_weights.csv',paired)
    directions=[]
    for row in rows:
        for model in protocol.VALIDATORS:
            part=[r for r in images if r['composition']==row['composition'] and r['pull']==row['pull'] and r['validator']==model]
            for tid in protocol.development_ids():
                d=[r for r in part if r['transition_id']==tid]
                directions.append(dict(composition=row['composition'],pull=row['pull'],validator=model,transition=tid,
                    hits=sum(r['target_hit'] for r in d) if len(d)==2 else None,denominator=2,observed=len(d)))
    scheduler.table(protocol.ROOT/'reports/directions.csv',directions)
    protocol.atomic_json(protocol.ROOT/'progress.json',dict(at=time.time(),conditions=rows,failures=state.get('failures',[]),
        active=state.get('active',[]),commercial_testing='AWAITING_USER_CONFIRMATION',full_matrix_expansion=False))
    lines=['# Does balanced Pull/Push generalize across proxy combinations?','',
        'Same frozen ten-image historical surrogate development cohort and three held-out checkpoints. Four layers, DI, no patch, BF16,50 steps. Ratio selection never reads commercial/final-target outputs.', '',
        '| Composition | Pull | Status | Target hits /30 model-image observations |','|---|---:|---|---|']
    lines += [f"| {r['composition']} | {r['pull']} | {r['status']} | {r['hits'] if r['hits'] is not None else 'N/A'}/30 |" for r in rows]
    lines += ['', 'Compare ratios within each composition using paired_weights.csv, and model/direction details in directions.csv. Observations from three validators on the same image are dependent; bootstrap resamples ten image clusters. Partial results are not full-cohort scores. Matched singles supply composition controls; no target-side method selection or commercial calls.']
    (protocol.ROOT/'reports/report.md').write_text('\n'.join(lines)+'\n')


scheduler.ROOT=protocol.ROOT
scheduler.freeze=protocol.freeze
scheduler.reuse_controls=protocol.reuse_controls
scheduler.validation_path=protocol.validation_path
scheduler.attack_status=protocol.attack_status
scheduler.WORKER_SCRIPT=protocol.EXP/'src/pp4_composition_worker.py'
scheduler.report=report

if __name__=='__main__':scheduler.main()
