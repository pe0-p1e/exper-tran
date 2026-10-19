import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from unified_research_plan import conditions, VALIDATORS
from unified_scheduler import next_research
from run_commercial_matrix import entries
from cached_reference_geometry import gram,cka_from_grams


def test_followup_conditions_use_only_surrogates_and_named_compute_budgets():
    cs=conditions()
    assert len(cs)==11 and len({c['id'] for c in cs})==11
    for c in cs:
        assert not set(c['models'])&set(VALIDATORS)
        assert c['steps']==50 and c['seed']==42 and c['dev_offset']==2
        assert c['images_per_direction']==2
        assert (c['budget_class']=='expanded_forward_budget')==(c['options'].get('eot_samples',1)>1)
    assert cs[1]['options']['loss']=='pull_push' and cs[1]['options']['patch_weight']==.35
    assert cs[2]['options']['patch_weight']==0


def test_research_dependencies_do_not_validate_before_attack_or_skip_embeddings():
    c=conditions()[0]
    names=sorted(set(c['models']+VALIDATORS))
    s=dict(attack_complete=False,validations={v:False for v in VALIDATORS},embeddings={n:False for n in names})
    assert next_research(c,s)==('attack',c['models'])
    s['attack_complete']=True
    assert next_research(c,s)==('validate',[VALIDATORS[0]])
    s['validations']={v:True for v in VALIDATORS}
    assert next_research(c,s)==('embeddings',[names[0]])
    s['embeddings']={n:True for n in names}
    assert next_research(c,s)==(None,[])


def test_model_matrix_rejects_paid_fallback_and_duplicate_conditions():
    path=Path(__file__).resolve().parents[1]/'commercial_model_matrix.json'
    config=json.loads(path.read_text())
    rows=entries(config)
    assert len(rows)==12
    bad=copy.deepcopy(config);bad['paid_api_fallback']=True
    with pytest.raises(ValueError):entries(bad)
    bad=copy.deepcopy(config);bad['models'].append(dict(rows[0],id='duplicate'))
    with pytest.raises(ValueError):entries(bad)


def test_sample_gram_cka_is_invariant_to_translation_and_rotation():
    rng=np.random.default_rng(1);x=rng.normal(size=(20,4));q,_=np.linalg.qr(rng.normal(size=(4,4)))
    assert cka_from_grams(gram(x),gram(3*x@q+7))==pytest.approx(1.)
    assert cka_from_grams(gram(x),gram(rng.normal(size=(20,4))))<.8
    assert cka_from_grams(gram(np.zeros((20,4))),gram(x)) is None
