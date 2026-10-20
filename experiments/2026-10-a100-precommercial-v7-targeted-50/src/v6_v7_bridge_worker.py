import pp4_worker as worker
from v6_v7_bridge_protocol import ROOT,VALIDATORS,plan,validation_path

def load_condition(cid):
    p=plan()
    return next(c for c in [r['condition'] for r in p['ladder']]+[p['smoke']] if c['id']==cid)
worker.ROOT=ROOT
worker.VALIDATORS=VALIDATORS
worker.load_condition=load_condition
worker.validation_path=validation_path
if __name__=='__main__':worker.main()
