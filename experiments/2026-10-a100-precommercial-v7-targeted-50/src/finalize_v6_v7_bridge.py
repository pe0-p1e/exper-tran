"""Paired small-bridge contrasts, completeness verification and compact archival."""
import json
import time
import numpy as np
import v6_v7_bridge_protocol as protocol
from pp4_campaign import table
import pp4_publish as publisher

if __name__=='__main__':
    root=protocol.ROOT
    while not (root/'completion.json').exists():time.sleep(15)
    observed={};checks=[]
    for row in protocol.plan()['ladder']:
        c=row['condition'];attack=protocol.attack_status(c,deep=True)
        path=protocol.validation_path(c,protocol.VALIDATORS[0])
        if attack['complete'] and path.exists():
            data=json.loads(path.read_text());items=data['images']
            if len(items)!=10 or len({r['image_id'] for r in items})!=10:raise RuntimeError('Wrong sample count')
            if any(r['target_hit']!=(r['adv_label']==r['target_label']) for r in items):raise RuntimeError('Wrong numerator')
            observed[row['label']]={r['image_id']:int(r['target_hit']) for r in items}
        checks.append(dict(label=row['label'],complete=row['label'] in observed))
    pairs=[]
    contrasts=[('layer_identity_noDI','old_deep_noDI','single75_noDI'),('DI_deep','old_deep_noDI','deep_DI'),
        ('DI_75','single75_noDI','single75_DI'),('DI_four','four_noDI','four_DI'),
        ('aggregation_with_DI','single75_DI','four_DI'),('coefficient','four_DI','four_DI_balanced'),
        ('whole_pipeline','old_deep_noDI','four_DI_balanced')]
    for factor,a,b in contrasts:
        if a not in observed or b not in observed:continue
        if observed[a].keys()!=observed[b].keys():raise RuntimeError('Different paired images')
        keys=sorted(observed[a]);d=np.array([observed[b][k]-observed[a][k] for k in keys])
        boot=np.random.default_rng(42).choice(d,size=(4000,10),replace=True).mean(1)
        pairs.append(dict(factor=factor,before=a,after=b,N=10,delta_ASR=float(d.mean()),
            CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975)),
            wins=int((d>0).sum()),losses=int((d<0).sum()),caveat='Exploratory inspected dev10; intervals do not establish equivalence; multiple contrasts'))
    table(root/'reports/paired_contrasts.csv',pairs)
    protocol.atomic_json(root/'reports/completeness_check.json',dict(conditions=checks,
        complete=all(c['complete'] for c in checks),no_final_target_inference=True,
        validation_model=protocol.VALIDATORS[0],same_V6_E11_proxy_order=protocol.MODELS))
    publisher.ROOT=root;publisher.DEST_FOLDER='v6_v7_multi_bridge';publisher.publish()
