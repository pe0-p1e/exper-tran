import time
import pp4_campaign as scheduler
import layer_selection_pilot_protocol as protocol

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
    (protocol.ROOT/'reports/report.md').write_text('# Small layer/model selection pilot\n\nTen predeclared conditions, same historical development images, Qwen9 held-out cross-family surrogate only. Fixed balanced Pull/Push, DI and equal per-image L1-normalized fusion. Six layer conditions retain E11 model order; four model-count controls retain four uniform layers. 50 steps,48 references/class, strict BF16. Not formal target results; selection bias and n=10 limit interpretation. The four-uniform matched baseline is reused only after source, cohort and PNG integrity checks. Non-quartile taps are exploratory controls, not a reproduction of ILA latest-peak or FDA greedy classifier selection.\n\n'+ '\n'.join(f"{r['label']}: {r['status']} {r['hits'] if r['hits'] is not None else 'N/A'}/10" for r in rows)+'\n')

scheduler.ROOT=protocol.ROOT
scheduler.VALIDATORS=protocol.VALIDATORS
scheduler.freeze=protocol.freeze
scheduler.reuse_controls=protocol.reuse_controls
scheduler.validation_path=protocol.validation_path
scheduler.attack_status=protocol.attack_status
scheduler.WORKER_SCRIPT=protocol.EXP/'src/layer_selection_pilot_worker.py'
scheduler.report=report
if __name__=='__main__':scheduler.main()
