"""No-model regression tests for frozen artifact identity and layer indexing."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vision_features import layer_indices
import run_v7_attack as runner
from v7_common import sha


def test_normalized_depth_matches_real_block_contract():
    assert layer_indices(24,[.01,.25,.5,.75,1.])==[0,5,11,17,23]
    assert layer_indices(16,[.01,.25,.5,.75,1.])==[0,3,7,11,15]


def test_resume_rejects_modified_or_missing_pngs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'OUT',tmp_path)
    png=tmp_path/'adv.png';png.write_bytes(b'original')
    state=tmp_path/'state.json'
    state.write_text(json.dumps(dict(status='attack_complete',identity='locked',png_hashes={'adv.png':sha(png)})))
    assert runner.valid_state(state,'locked')
    assert runner.valid_state(state,'different-method') is None
    png.write_bytes(b'tampered')
    assert runner.valid_state(state,'locked') is None
    png.unlink()
    assert runner.valid_state(state,'locked') is None


def test_reference_only_transforms_have_finite_image_gradients():
    import torch
    bank=torch.randn(10,48,6)
    for mode in ['none','center','standardize']:
        x=torch.randn(2,6,requires_grad=True)
        scores=runner.feature_scores(x,bank,dict(feature_transform=mode,aggregation='centroid'))
        grad=torch.autograd.grad(scores[:,1].sum(),x)[0]
        assert torch.isfinite(scores).all() and torch.isfinite(grad).all()
        assert grad.abs().sum()>0


def test_spectral_view_reproducible_and_differentiable():
    import torch
    x=torch.rand(2,3,16,16,requires_grad=True)
    a=runner.frequency_view(x,.15,torch.Generator().manual_seed(42))
    b=runner.frequency_view(x,.15,torch.Generator().manual_seed(42))
    assert torch.equal(a,b)
    assert a.min()>=0 and a.max()<=1
    grad=torch.autograd.grad(a.square().sum(),x)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
