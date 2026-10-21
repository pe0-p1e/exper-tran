"""Nested balanced30-of50 confirmation; never alter the scientific cohort or RNG."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C

def ordered_cases(cases):
    if len(cases)!=50 or len({r['image_id'] for r in cases})!=50:raise ValueError('Expected50 distinct confirmation images')
    groups={t:[r for r in cases if r['transition']==t] for t in C.TIDS}
    if any(len(v)!=10 for v in groups.values()):raise ValueError('Expected10 images per direction')
    return [r for start,end in [(0,2),(2,6),(6,10)] for t in C.TIDS for r in groups[t][start:end]]
def cases(count=50):
    if count not in [30,50]:raise ValueError('Only predeclared30/50 confirmation')
    return ordered_cases(C.cohort('confirm')['images'])[:count]
def frozen(c,count=30):
    if c['split']!='confirm':return False
    lock=C.read(C.ROOT/'method_lock.json')
    if not lock or c['family'] not in lock['families']:return False
    family=lock['families'][c['family']]
    if c['layers'] not in [family['primary'],family['uniform']]:return False
    C.legal(c['layers'],c['validator'])
    for r in cases(count):
        p=C.image_receipt(c,r['image_id'])
        if not C.valid_receipt(p,c):return False
        receipt=C.read(p)
        if any(receipt.get(k)!=r[k] for k in ['image_id','source','target','transition']):raise RuntimeError('Wrong confirmation image/label')
    return True
def path(c):return C.ROOT/'validation30'/(c['id']+'.json')
def done(c):
    v=C.read(path(c))
    return bool(v and v.get('complete') and v.get('science')==C.digest(C.scientific(c)) and v.get('N')==30
      and {x['image_id'] for x in v.get('images',[])}=={r['image_id'] for r in cases(30)})
def pending(c):
    result=[]
    if frozen(c,30) and not done(c):result.append((c,'evaluate30'))
    if not C.attack_done(c):result.append((c,'attack'))
    elif done(c) and not C.eval_done(c):result.append((c,'evaluate'))
    return result

def summarize():
    rows=[]
    for p in (C.ROOT/'validation30').glob('*.json'):
        v=C.read(p);c=C.read(C.ROOT/'conditions'/(v['condition']+'.json'))
        if not c or not done(c):continue
        rows.append(dict(condition=c['id'],family=c['family'],model=v['model'],hits=v['hits'],N=30,ASR=v['hits']/30,
          directions=[dict(transition=t,hits=sum(x['target_hit'] for x in v['images'] if x['transition']==t),N=6) for t in C.TIDS]))
    result=dict(complete_conditions=len(rows),rows=rows,scope='nested30_of_frozen50_confirmation_not_development30',commercial=C.read(C.ROOT/'stage2/summary30.json',{}))
    C.atomic(C.ROOT/'reports/confirmation30.json',result)
    return result
