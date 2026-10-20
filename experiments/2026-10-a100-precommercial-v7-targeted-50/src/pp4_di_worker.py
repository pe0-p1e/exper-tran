"""Use frozen PP4 attack and validator without changing scientific sources."""
import pp4_worker as worker
from pp4_di_protocol import ROOT,plan,validation_path

def load_condition(cid):
    p=plan()
    return next(c for c in [r['condition'] for r in p['ladder']]+[p['smoke']] if c['id']==cid)

worker.ROOT=ROOT
worker.validation_path=validation_path
worker.load_condition=load_condition
if __name__=='__main__':worker.main()
