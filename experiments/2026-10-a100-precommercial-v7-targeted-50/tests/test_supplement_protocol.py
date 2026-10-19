import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supplement_protocol import candidates,candidate_key
from supplement_registry import registry,native_clip_preprocess
from run_multiproxy_supplement import admission
from family_protocol import require_excluded_family
from v7_common import REGISTRY,EXP,sha


def test_matrix_is_exact_union_with_correct_original_selection_overlap():
    rows=candidates()
    assert len(rows)==47
    assert sum(len(r['aliases']) for r in rows)==52
    original={a:r for r in rows for a in r['aliases'] if a.startswith('E')}
    assert set(original)=={f'E{i:02d}' for i in range(1,14)}
    assert original['E10']['config']['id']=='PSQ_AB'
    assert original['E11']['source']=='corrected_family_disjoint'
    assert original['E04']['source']=='historical_valid_Gemma_cross_family'
    required=[r for r in rows if not r['optional'] and len(r['config']['models'])>1 and r['source'] not in ('corrected_family_disjoint','historical_valid_Gemma_cross_family')]
    assert len(required)==17
    assert sum(r['optional'] for r in rows)==18


def test_every_candidate_excludes_entire_target_family_and_repeated_weights():
    reg=registry()
    for r in candidates():
        c=r['config'];require_excluded_family(c['models'],reg[c['target']]['family'],reg)
    with pytest.raises(ValueError,match='Repeated'):candidate_key('target',['same','same'])
    with pytest.raises(ValueError,match='Forbidden'):require_excluded_family(['Qwen3.5-2B'],'qwen',reg)


def test_extension_preserves_base_pins_and_six_distinct_visual_checkpoints():
    before=copy.deepcopy(REGISTRY);reg=registry()
    assert REGISTRY==before
    assert all(reg[n]==s for n,s in before.items())
    names=['CLIP ViT-B/32','CLIP ViT-B/16','CLIP ViT-L/14','CLIP ViT-L/14@336','DINOv2-Base','DINOv2-Large']
    assert len({reg[n]['repo_id'] for n in names})==6
    assert all(len(reg[n]['revision'])==40 for n in names)
    assert reg['CLIP ViT-L/14@336']['native_image_size']==336
    assert reg['DINOv2-Base']['native_image_size']==256
    assert reg['DINOv2-Base']['config_image_size']==518


def test_native_clip_resize_keeps_gradient_to_224_canvas():
    x=torch.rand(1,3,224,224,requires_grad=True)
    out=native_clip_preprocess(x,336,[.48,.45,.4],[.26,.26,.27])
    assert out.shape==(1,3,336,336) and out.dtype==torch.bfloat16
    gradient=torch.autograd.grad(out.float().square().mean(),x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_budget_reserves_evaluation_and_validation_instead_of_only_attack():
    assert not admission(2000,1000)
    assert admission(2800,1000)
    assert not admission(1799,0)


def test_shared_worker_yields_before_primary_target_loading_and_limits_memory():
    from run_parallel_subsets import worker_guard
    assert worker_guard(dict(stage='attack'),dict(used_gib=20,worker_count=2),False) is None
    assert worker_guard(dict(stage='target_evaluation'),dict(used_gib=20),False)
    assert worker_guard(dict(stage='attack'),dict(used_gib=73),False)
    assert worker_guard(dict(stage='attack'),dict(used_gib=20,worker_count=5),False)
    assert worker_guard(dict(stage='target_evaluation'),dict(used_gib=60),True) is None


def test_child_registry_install_cannot_change_parent_or_frozen_files():
    paths=[EXP/'src/vision_features.py',EXP.parent/'2026-10-a100-precommercial-v6/model_revisions.json']
    before=[sha(p) for p in paths]
    cmd=[sys.executable,'-c',f"import sys;sys.path.insert(0,{str(EXP/'src')!r});from supplement_registry import install; r=install();assert 'CLIP ViT-B/32' in r"]
    subprocess.run(cmd,check=True)
    assert 'CLIP ViT-B/32' not in REGISTRY
    assert [sha(p) for p in paths]==before
