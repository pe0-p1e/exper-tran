from v6_v7_bridge_protocol import plan,MODELS,VALIDATORS

def test_bridge_changes_settings_not_ensemble_or_evaluator():
    p=plan();assert len(p['ladder'])==7 and VALIDATORS==['Qwen3.5-9B']
    cs={r['label']:r['condition'] for r in p['ladder']}
    assert len({c['id'] for c in cs.values()})==7
    for c in cs.values():
        assert c['models']==MODELS and c['steps']==50 and c['split']=='dev'
        assert c['options']['patch_weight']==0 and c['options']['loss']=='pull_push'
        assert 'Qwen3.5-2B' not in c['models']
    for a,b in [('old_deep_noDI','deep_DI'),('single75_noDI','single75_DI'),('four_noDI','four_DI')]:
        assert cs[a]['options']==cs[b]['options'] and not cs[a]['di'] and cs[b]['di']
    a,b=cs['four_DI']['options'],cs['four_DI_balanced']['options']
    assert {k:v for k,v in a.items() if k not in ['pull_weight','push_weight']}=={k:v for k,v in b.items() if k not in ['pull_weight','push_weight']}
