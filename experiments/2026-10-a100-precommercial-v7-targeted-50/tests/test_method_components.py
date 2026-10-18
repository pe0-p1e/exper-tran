"""CPU unit tests for V7 surrogate-only attack components.

Run with:
pytest -q experiments/2026-10-a100-precommercial-v7-targeted-50/tests/
Tests do NOT invoke target-model weights or APIs.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from representation_objectives import (
    all_class_margin_loss,
    class_scores,
    combine_multi_layer,
    local_patch_target_loss,
    original_pull_push_loss,
)
from attack_optimizers import (
    adaptive_weights_from_surrogate_lookahead,
    differentiable_resize_pad,
    fuse_proxy_gradients,
    normalize_pixel_gradients,
)


def _bank():
    torch.manual_seed(3)
    return torch.randn(10, 5, 6)


def test_scores_shapes_and_derivatives():
    bank = _bank()
    images = torch.randn(4, 6, requires_grad=True)
    for mode in ["centroid", "soft_reference"]:
        scores = class_scores(images, bank, aggregation=mode)
        assert scores.shape == (4, 10)
        loss = all_class_margin_loss(scores, 4)
        grad = torch.autograd.grad(loss, images, retain_graph=True)[0]
        assert bool(torch.isfinite(grad).all())
        assert float(grad.abs().max()) > 0


def test_original_loss_and_multilayer():
    x = torch.randn(4, 6, requires_grad=True)
    bank = _bank()
    a = original_pull_push_loss(x, bank, source_class=1, target_class=2)
    b = original_pull_push_loss(x, bank, source_class=1, target_class=3)
    combined = combine_multi_layer({"shallow": a, "deep": b}, {"shallow": 1, "deep": 3})
    assert torch.allclose(combined, (a + 3*b)/4)
    assert torch.isfinite(torch.autograd.grad(combined, x)[0]).all()


def test_local_patch_loss_and_gradients():
    x = torch.randn(2, 16, 8, requires_grad=True)
    tar = torch.randn(5, 8)
    src = torch.randn(5, 8)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 0] = False
    loss = local_patch_target_loss(x, tar, src, patch_mask=mask)
    grad = torch.autograd.grad(loss, x)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(grad).all()
    assert grad[:, 0].abs().max() == 0


def test_gradient_fusion_weighting_and_conflicts():
    gradients = torch.randn(3, 2, 3, 8, 8)
    g = normalize_pixel_gradients(gradients)
    assert torch.allclose(g.abs().flatten(2).sum(-1), torch.ones(3, 2), atol=1e-6)
    equal = fuse_proxy_gradients(gradients, mode="equal")
    assert torch.allclose(equal, g.mean(dim=0))
    one_hot = fuse_proxy_gradients(gradients, mode="weighted", weights=torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(one_hot, g[0])
    conflict = fuse_proxy_gradients(gradients, mode="pcgrad")
    assert torch.isfinite(conflict).all()


def test_adaptive_weights_use_only_heldout_surrogates():
    improvements = torch.zeros(3, 3, 2)
    improvements[0, 1, :] = 0.6
    improvements[0, 2, :] = 0.5
    weights = adaptive_weights_from_surrogate_lookahead(improvements)
    assert weights.shape == (3, 2)
    assert (weights[0] > weights[1]).all()
    assert torch.allclose(weights.sum(0), torch.ones(2))


def test_input_diversity_preserves_gradient_path_and_canvas():
    x = torch.rand(2, 3, 224, 224, requires_grad=True)
    rng = torch.Generator().manual_seed(42)
    transformed = differentiable_resize_pad(x, generator=rng)
    assert transformed.shape == x.shape
    transformed.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        all_class_margin_loss(torch.zeros(2,10), 0)
    with pytest.raises(ValueError):
        local_patch_target_loss(torch.zeros(2,4,8), torch.zeros(4,7), torch.zeros(4,7))
    with pytest.raises(ValueError):
        fuse_proxy_gradients(torch.randn(2, 1, 3, 4, 4), mode="weighted", weights=torch.zeros(2))


def test_margin_descent_improves_target_against_all_rivals():
    # Orthogonal reference classes make the intended direction unambiguous.
    refs=torch.eye(10)[:,None,:].repeat(1,3,1).to(torch.bfloat16)
    z=torch.zeros(1,10);z[0,0]=1;z.requires_grad_(True)
    scores=class_scores(z,refs)
    loss=all_class_margin_loss(scores,2)
    grad=torch.autograd.grad(loss,z)[0]
    after=class_scores(z-.1*grad,refs)
    assert after[0,1]-after[0,0] > scores[0,1]-scores[0,0]
    assert all_class_margin_loss(after,2) < loss


def test_opposing_proxy_gradients_are_not_artificially_amplified():
    gradients=torch.tensor([1.,-1.]).reshape(2,1,1,1,1).expand(2,1,3,2,2)
    fused=fuse_proxy_gradients(gradients,mode='pcgrad')
    assert torch.allclose(fused,torch.zeros_like(fused))


def test_patch_per_image_reduction_matches_mean_and_ignores_padding():
    x=torch.randn(3,7,5,requires_grad=True)
    mask=torch.ones(3,7,dtype=torch.bool);mask[:,-1]=False
    target=torch.randn(4,5);source=torch.randn(4,5)
    per=local_patch_target_loss(x,target,source,patch_mask=mask,reduction='none')
    mean=local_patch_target_loss(x,target,source,patch_mask=mask)
    assert per.shape==(3,)
    assert torch.allclose(per.mean(),mean)
    grad=torch.autograd.grad(per.sum(),x)[0]
    assert grad[:,-1].abs().max()==0
