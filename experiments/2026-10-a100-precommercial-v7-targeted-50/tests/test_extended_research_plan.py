"""Validate extension/resume semantics and controlled mechanism comparisons."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from extended_research_plan import extended_budget,layer_conditions,complementarity_conditions,unique_candidates


def test_extension_is_total_hours_not_another_twenty_on_resume():
    first=dict(started=100,deadline=36100,hours=10)
    extended=extended_budget(first,20,now=1000)
    assert extended==dict(started=100,deadline=72100,hours=20)
    assert extended_budget(extended,20,now=7000)==extended
    assert extended_budget(extended,10,now=7000)==extended
    assert first['deadline']==36100
    with pytest.raises(ValueError):extended_budget(first,float('nan'))


def test_layer_removal_keeps_models_loss_and_other_depths_fixed():
    models=['p1','p2','p3'];conditions=layer_conditions(models)
    assert len(conditions)==14
    assert len({x['id'] for x in conditions})==14
    for i in range(4):
        row=next(x for x in conditions if x['id']==f'T20_layer_without_{i}')
        assert row['options']['layers']==[d for j,d in enumerate([.25,.5,.75,1.]) if j!=i]
    assert all(x['models']==models and x['method']=='A3' for x in conditions)
    assert all('seed' not in x and 'steps' not in x for x in conditions)


def test_proxy_controls_preserve_loss_and_every_member_gets_single():
    anchor=dict(id='old',models=['a','b','c'],method='A3',options={'temperature':.02},fusion='pcgrad')
    rows=complementarity_conditions(anchor)
    assert {tuple(x['models']) for x in rows if len(x['models'])==1}=={('a',),('b',),('c',)}
    assert {tuple(x['models']) for x in rows if len(x['models'])==2}=={('a','b'),('a','c'),('b','c')}
    assert all(x['options']==anchor['options'] for x in rows)
    rows[0]['options']['temperature']=1
    assert anchor['options']['temperature']==.02


def test_confirmation_ignores_repeated_identical_controls():
    rows=[{'config':dict(id='one',models=['a','b'],method='A3')},
          {'config':dict(id='repeat',models=['a','b'],method='A3',di=True,fusion='equal')},
          {'config':dict(id='different',models=['a','b'],method='A3',di=False)}]
    assert [x['config']['id'] for x in unique_candidates(rows)]==['one','different']
