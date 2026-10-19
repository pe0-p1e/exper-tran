import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from family_protocol import require_excluded_family,require_condition_target,corrected_conditions,condition_name
from v7_common import REGISTRY


def test_all_corrected_conditions_are_family_disjoint_and_match_full_budget():
    rows=corrected_conditions()
    assert len(rows)==32
    assert len({condition_name(c) for c in rows})==32
    for c in rows:
        require_excluded_family(c['models'],REGISTRY[c['target']]['family'],REGISTRY)
        assert c['steps']==50 and c['restarts']==1
    assert {c['target'] for c in rows}=={'Qwen3.5-27B','InternVL3.5-14B-HF'}


def test_family_gate_rejects_all_same_family_sizes():
    for name in ['Qwen3.5-2B','Qwen3.5-4B','Qwen3.5-9B']:
        with pytest.raises(ValueError,match='Forbidden target-family'):
            require_excluded_family(['CLIP ViT-L/14',name],'qwen',REGISTRY)
    with pytest.raises(ValueError):require_excluded_family(['InternVL3.5-2B-HF'],'internvl',REGISTRY)


def test_evaluator_rejects_wrong_target_and_proxy_substitution():
    c=corrected_conditions()[0];name=condition_name(c)
    lock=dict(protocol='strict_family_disjoint',condition_targets={name:c['target']},condition_models={name:c['models']})
    require_condition_target(name,c['target'],c['models'],lock,REGISTRY)
    with pytest.raises(ValueError):require_condition_target(name,'InternVL3.5-14B-HF',c['models'],lock,REGISTRY)
    with pytest.raises(ValueError):require_condition_target(name,c['target'],['CLIP ViT-L/14'],lock,REGISTRY)
