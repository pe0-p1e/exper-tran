import pytest
import torch
from pp4_raw_fusion import raw_equal_mean
from attack_optimizers import fuse_proxy_gradients

def test_raw_control_exposes_magnitude_dominance():
    g=torch.tensor([100.,-1.]).reshape(2,1,1,1,1)
    assert raw_equal_mean(g).item()==49.5
    assert fuse_proxy_gradients(g).item()==0.
    assert torch.equal(g.flatten(),torch.tensor([100.,-1.]))
    with pytest.raises(ValueError):raw_equal_mean(g,mode='pcgrad')

def test_pilot_removes_only_declared_component():
    from pp4_mechanism_protocol import plan
    rows=plan()['ladder']
    assert len(rows)==10 and len({r['condition']['id'] for r in rows})==10
    for group in ['A','B']:
        cs={r['label']:r['condition'] for r in rows if r['group']==group}
        base=cs['reference']
        for label,c in cs.items():
            assert c['models']==base['models'] and c['di'] and c['steps']==50
            assert c['options']['pull_weight']==c['options']['push_weight']==.5
            if label=='raw_gradient_mean':
                assert {k:v for k,v in c['options'].items() if k not in ['gradient_normalization','raw_fusion_source_sha256']}==base['options']
            else:
                assert {k:v for k,v in c['options'].items() if k!='layers'}=={k:v for k,v in base['options'].items() if k!='layers'}
