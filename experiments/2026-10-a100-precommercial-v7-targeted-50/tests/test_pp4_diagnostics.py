import pytest
import torch
from pp4_objective import objective
from representation_objectives import original_pull_push_loss


class Proxy:
    def __init__(self):
        rng=torch.Generator().manual_seed(13)
        self.weights={l:torch.randn(3,5,generator=rng) for l in range(4)}
        self.bank={l:(torch.randn(10,4,5,generator=rng),None) for l in range(4)}

    def features(self,x):
        return {l:(x@w,None,None) for l,w in self.weights.items()}


def cfg(pull):
    return dict(loss='pull_push',patch_weight=0.,projector_weight=0.,
                pull_weight=pull,push_weight=1-pull,layers=[.25,.5,.75,1.])


def test_all_four_layers_and_weighted_gradient():
    proxy=Proxy();x=torch.tensor([[.2,.4,.6],[.8,.3,.1]],requires_grad=True)
    actual,_=objective(proxy,x,1,2,cfg(.75))
    expected=torch.stack([original_pull_push_loss(x@proxy.weights[l],proxy.bank[l][0],1,2,reduction='none') for l in range(4)]).mean(0)
    assert torch.allclose(actual,expected)
    g=torch.autograd.grad(actual.sum(),x,retain_graph=True)[0]
    pure_pull=objective(proxy,x,1,2,cfg(1))[0]
    pure_push=objective(proxy,x,1,2,cfg(0))[0]
    independent=.75*torch.autograd.grad(pure_pull.sum(),x,retain_graph=True)[0]+.25*torch.autograd.grad(pure_push.sum(),x)[0]
    assert torch.allclose(g,independent,atol=1e-6)
    assert torch.isfinite(g).all() and (g.abs().sum(1)>0).all()


def test_pure_objective_rejects_unrequested_terms():
    for options in [dict(patch_weight=.35),dict(projector_weight=.2),dict(loss='margin')]:
        with pytest.raises(ValueError):objective(Proxy(),torch.ones(1,3),1,2,{**cfg(.75),**options})


def test_bounded_plan_has_no_targets_and_no_full_matrix():
    from pp4_protocol import plan,VALIDATORS
    p=plan();rows=p['ladder']+p['weights']
    assert set(p)=={'ladder','weights','smoke'}
    assert len({r['condition']['id'] for r in rows})==14
    assert all(not(set(r['condition']['models'])&set(VALIDATORS)) for r in rows)
    assert all(r['condition']['split']=='dev' for r in rows)
    assert all(r['condition']['steps']==50 for r in rows)


def test_composition_ratio_pairs_preserve_every_other_setting():
    from pp4_composition_plan import plan,COMBINATIONS,VALIDATORS
    rows=plan()['ladder']
    assert len(rows)==26 and len(COMBINATIONS)==13
    assert len({r['condition']['id'] for r in rows})==26
    for group,models in COMBINATIONS:
        pair=[r['condition'] for r in rows if r['group']==group]
        assert {c['options']['pull_weight'] for c in pair}=={.5,.75}
        assert all(c['models']==models and not(set(models)&set(VALIDATORS)) for c in pair)
        a,b=[{k:v for k,v in c['options'].items() if k not in ['pull_weight','push_weight']} for c in pair]
        assert a==b
        assert all(c['options']['patch_weight']==0 and c['di'] and c['steps']==50 for c in pair)
