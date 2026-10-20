import time
import numpy as np
import pp4_campaign as scheduler
import pp4_mechanism_protocol as protocol

def report(p,state):
    rows=[];images=[];observed={}
    for row in p['ladder']:
        c=row['condition'];a,v=scheduler.evidence(c);done=a['complete'] and len(v)==len(protocol.VALIDATORS)
        obs=[dict(group=row['group'],label=row['label'],validator=m,**r) for m,d in v.items() for r in d['images']];images+=obs
        rows.append(dict(group=row['group'],label=row['label'],condition=c['id'],status='COMPLETE' if done else 'PARTIAL' if a['N'] else 'QUEUED',
            attack_images=a['N'],observed=len(obs),hits=sum(r['target_hit'] for r in obs) if done else None,
            denominator=30,unique_images=10,reused=bool(a.get('reused')),attack_seconds=a['seconds']))
        if done:observed[(row['group'],row['label'])]={(r['validator'],r['image_id']):r for r in obs}
    pairs=[]
    for (group,label),right in observed.items():
        if label=='reference' or (group,'reference') not in observed:continue
        left=observed[(group,'reference')]
        if left.keys()!=right.keys():raise RuntimeError('Pair identities differ')
        ids=sorted({k[1] for k in left})
        delta=np.array([np.mean([int(right[k]['target_hit'])-int(left[k]['target_hit']) for k in left if k[1]==iid]) for iid in ids])
        boot=np.random.default_rng(42).choice(delta,size=(4000,len(ids)),replace=True).mean(1)
        pairs.append(dict(group=group,label=label,delta_ASR=float(delta.mean()),unique_images=10,
            CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975))))
    for name,data in [('conditions',rows),('per_image',images),('paired',pairs)]:scheduler.table(protocol.ROOT/'reports'/(name+'.csv'),data)
    protocol.atomic_json(protocol.ROOT/'progress.json',dict(at=time.time(),conditions=rows,active=state.get('active',[]),failures=state.get('failures',[]),full_matrix=False,commercial_testing=False))
    (protocol.ROOT/'reports/report.md').write_text('# Small mechanism pilots\n\nTwo fixed compositions, ten previously inspected dev images each, three held-out checkpoints. Same image observations are correlated. No final-target or commercial inference. DI always on; only fusion normalization OR layer subset changes. Full reference scores reused, no exhaustive sweep.\n\n'+ '\n'.join(f"{r['group']} {r['label']}: {r['status']}, {r['hits'] if r['hits'] is not None else 'N/A'}/30" for r in rows)+'\n')

scheduler.ROOT=protocol.ROOT
scheduler.freeze=protocol.freeze
scheduler.reuse_controls=protocol.reuse_controls
scheduler.validation_path=protocol.validation_path
scheduler.attack_status=protocol.attack_status
scheduler.WORKER_SCRIPT=protocol.EXP/'src/pp4_mechanism_worker.py'
scheduler.report=report
if __name__=='__main__':scheduler.main()
