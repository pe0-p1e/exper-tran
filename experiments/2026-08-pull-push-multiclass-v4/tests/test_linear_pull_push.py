import pytest
import torch
import torch.nn.functional as F

from primary_ml_cka.attack.losses.semantic_contrastive import semantic_representation_loss


def compute(x, mode, pull=1., push=.5, tau=.1):
    return semantic_representation_loss(
        x, torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]]),
        mode=mode, target_logit_weight=pull, source_logit_weight=push, tau=tau,
    )


def test_linear_components_and_temperature_invariance():
    x = torch.tensor([[.3, .6, .7]], requires_grad=True)
    out = compute(x, "linear_pull_push")
    unit = F.normalize(x, dim=-1)
    torch.testing.assert_close(out.loss, (1 - unit[:, 0] + .5 * (1 + unit[:, 1])).mean())
    torch.testing.assert_close(out.loss, compute(x, "linear_pull_push", tau=.5).loss)


@pytest.mark.parametrize("pull,push", [(1., 0.), (0., 1.), (1., .5)])
def test_step_moves_requested_components(pull, push):
    x = torch.tensor([[.3, .4, .8]], requires_grad=True)
    before = compute(x, "linear_pull_push", pull, push)
    grad = torch.autograd.grad(before.loss, x)[0]
    after = compute(x - .01 * grad, "linear_pull_push", pull, push)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    if pull:
        assert after.target_similarity > before.target_similarity
    if push:
        assert after.source_similarity < before.source_similarity


def test_single_image_infonce_gradient_is_positive_rescaling():
    x = torch.tensor([[.3, .4, .8]], requires_grad=True)
    linear = compute(x, "linear_pull_push")
    nce = compute(x, "prototype")
    gl = torch.autograd.grad(linear.loss, x, retain_graph=True)[0]
    gn = torch.autograd.grad(nce.loss, x)[0]
    factor = torch.sigmoid((.5 * linear.source_similarity - linear.target_similarity) / .1) / .1
    torch.testing.assert_close(gn, factor * gl)


def test_linear_pull_only_matches_target_only_baseline():
    x = torch.tensor([[.3, .4, .8]], requires_grad=True)
    torch.testing.assert_close(
        compute(x, "linear_pull_push", push=0).loss,
        compute(x, "target_only").loss,
    )
