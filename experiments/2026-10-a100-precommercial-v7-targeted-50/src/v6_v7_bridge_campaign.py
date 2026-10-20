import time
import pp4_campaign as scheduler
import v6_v7_bridge_protocol as protocol

def report(p,state):
    rows=[];images=[]
    for row in p['ladder']:
        c=row['condition'];a,v=scheduler.evidence(c);done=a['complete'] and len(v)==1
        obs=[dict(label=row['label'],validator=m,**r) for m,d in v.items() for r in d['images']];images+=obs
        rows.append(dict(label=row['label'],condition=c['id'],status='COMPLETE' if done else 'PARTIAL' if a['N'] else 'QUEUED',
            attack_images=a['N'],observed=len(obs),hits=sum(r['target_hit'] for r in obs) if done else None,denominator=10,
            layers=str(c['options']['layers']),DI=c['di'],pull=c['options']['pull_weight'],attack_seconds=a['seconds']))
    scheduler.table(protocol.ROOT/'reports/conditions.csv',rows);scheduler.table(protocol.ROOT/'reports/per_image.csv',images)
    protocol.atomic_json(protocol.ROOT/'progress.json',dict(at=time.time(),conditions=rows,active=state.get('active',[]),failures=state.get('failures',[]),full_matrix=False,final_target_access=False))
    (protocol.ROOT/'reports/report.md').write_text('# V6 E11 matched multi-proxy bridge\n\nExact original model order Intern2B+GemmaE2B+CLIP. Same ten dev images and Qwen9 cross-family held-out validator. No final Qwen27 target inference. Pure Pull/Push, BF16,50 steps; only declared layer/DI/ratio factors vary. Current-runner replay of old settings, not full old binary replay. Small exploratory counts cannot establish population-level equivalence.\n\n'+ '\n'.join(f"{r['label']}: {r['status']} {r['hits'] if r['hits'] is not None else 'N/A'}/10" for r in rows)+'\n')

scheduler.ROOT=protocol.ROOT
scheduler.VALIDATORS=protocol.VALIDATORS
scheduler.freeze=protocol.freeze
scheduler.reuse_controls=protocol.reuse_controls
scheduler.validation_path=protocol.validation_path
scheduler.attack_status=protocol.attack_status
scheduler.WORKER_SCRIPT=protocol.EXP/'src/v6_v7_bridge_worker.py'
scheduler.report=report
if __name__=='__main__':scheduler.main()
