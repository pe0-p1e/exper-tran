"""Bounded DI comparison using existing GPU scheduler; no full-matrix expansion."""
import time
import numpy as np
import pp4_campaign as scheduler
import pp4_di_protocol as protocol


def report(p,state):
    rows=[];images=[];complete={}
    for row in p['ladder']:
        c=row['condition'];a,v=scheduler.evidence(c);done=a['complete'] and len(v)==len(protocol.VALIDATORS)
        obs=[dict(group=row['group'],DI=c['di'],validator=m,**r) for m,d in v.items() for r in d['images']]
        images+=obs
        rows.append(dict(group=row['group'],condition=c['id'],DI=c['di'],status='COMPLETE' if done else 'PARTIAL' if a['N'] else 'QUEUED',
            attack_images=a['N'],observed=len(obs),hits=sum(r['target_hit'] for r in obs) if done else None,
            denominator=30,unique_images=10,reused=bool(a.get('reused')),attack_seconds=a['seconds']))
        if done:complete[(row['group'],c['di'])]={(r['validator'],r['image_id']):r for r in obs}
    pairs=[]
    for group in ['A','B']:
        if not all((group,flag) in complete for flag in [False,True]):continue
        on,off=complete[(group,True)],complete[(group,False)]
        if on.keys()!=off.keys():raise RuntimeError('Paired identities differ')
        ids=sorted({k[1] for k in on})
        delta=np.array([np.mean([int(off[k]['target_hit'])-int(on[k]['target_hit']) for k in on if k[1]==iid]) for iid in ids])
        boot=np.random.default_rng(42).choice(delta,size=(4000,len(ids)),replace=True).mean(1)
        pairs.append(dict(group=group,DI_on_hits=sum(r['target_hit'] for r in on.values()),DI_off_hits=sum(r['target_hit'] for r in off.values()),
            N_unique_images=10,N_model_image_pairs=30,off_minus_on=float(delta.mean()),
            image_cluster_CI_low=float(np.quantile(boot,.025)),image_cluster_CI_high=float(np.quantile(boot,.975)),
            removal_wins=sum(off[k]['target_hit'] and not on[k]['target_hit'] for k in on),
            removal_losses=sum(on[k]['target_hit'] and not off[k]['target_hit'] for k in on)))
    scheduler.table(protocol.ROOT/'reports/conditions.csv',rows)
    scheduler.table(protocol.ROOT/'reports/per_image.csv',images)
    scheduler.table(protocol.ROOT/'reports/paired_DI.csv',pairs)
    protocol.atomic_json(protocol.ROOT/'progress.json',dict(at=time.time(),conditions=rows,failures=state.get('failures',[]),active=state.get('active',[]),full_matrix=False,commercial_testing=False))
    lines=['# Small matched DI removal study','',
        'Two fixed compositions, ten historical development images, three held-out model checkpoints. Four-layer pure PP.5/.5, BF16,50 steps. Same images are correlated across validators; this is exploratory. DI-on reused after PNG/protocol checks. No final-target or commercial calls.','',
        '| Group | DI | Status | Hits /30 model-image observations |','|---|---|---|---|']
    lines += [f"| {r['group']} | {r['DI']} | {r['status']} | {r['hits'] if r['hits'] is not None else 'N/A'}/30 |" for r in rows]
    lines += ['', 'Paired DI-off minus DI-on intervals resample ten image clusters; no equivalence claim from nonsignificance. Saved attack traces retain per-proxy optimization and gradient compatibility. Full target geometry is not computed by this pilot. No automatic full-cohort expansion.']
    (protocol.ROOT/'reports/report.md').write_text('\n'.join(lines)+'\n')

scheduler.ROOT=protocol.ROOT
scheduler.freeze=protocol.freeze
scheduler.reuse_controls=protocol.reuse_controls
scheduler.validation_path=protocol.validation_path
scheduler.attack_status=protocol.attack_status
scheduler.WORKER_SCRIPT=protocol.EXP/'src/pp4_di_worker.py'
scheduler.report=report
if __name__=='__main__':scheduler.main()
