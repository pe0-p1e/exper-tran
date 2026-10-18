"""V7 transfer-attack gradient fusion and differentiable input diversity.

Only operates on gradients of surrogate visual representation losses. Target
model outputs must not be used to set weights or select updates.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def normalize_pixel_gradients(grads: Tensor) -> Tensor:
    """[K,B,C,H,W] -> per-proxy, per-image L1 normalized float32 gradients."""
    if grads.ndim != 5:
        raise ValueError("Expected [proxy,batch,channels,height,width] gradient tensor")
    g = grads.float()
    divisor = g.abs().flatten(start_dim=2).sum(-1).clamp_min(1e-12)
    return g / divisor[..., None, None, None]


def pcgrad_pixel_fusion(
    normalized_grads: Tensor, *, eps: float = 1e-12
) -> Tensor:
    """Project pairwise conflicting per-image gradients using ORIGINAL g_j.

    Uses a fixed order to make experiments reproducible. This is an
    *inspired* adaptation of PCGrad, not a reproduction of task-learning PCGrad.
    Returns projected per-proxy gradients [K,B,C,H,W] so they can be weighted.
    """
    if normalized_grads.ndim != 5:
        raise ValueError("Expected [K,B,C,H,W]")
    k = normalized_grads.shape[0]
    original = normalized_grads.float()
    updated = original.clone()
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            gi = updated[i].flatten(1)
            gj = original[j].flatten(1)
            dot = (gi * gj).sum(-1)
            norm2 = gj.square().sum(-1).clamp_min(eps)
            correction = (dot.clamp(max=0) / norm2)[:, None, None, None]
            updated[i] = updated[i] - correction * original[j]
    return updated


def fuse_proxy_gradients(
    gradients: Tensor,
    *,
    mode: str = "equal",
    weights: Tensor | None = None,
    consensus_threshold: float = 0.0,
) -> Tensor:
    """Combine surrogate image-gradients into one [B,C,H,W] update direction.

    Supported modes: equal, weighted, pcgrad, weighted_pcgrad, consensus.
    Weights are [K] or [K,B] surrogate-derived nonnegative coefficients;
    the runner must log their source, computational cost and every update.
    """
    g = normalize_pixel_gradients(gradients)
    k, b = g.shape[:2]
    if mode in {"pcgrad", "weighted_pcgrad"}:
        g = pcgrad_pixel_fusion(g)
    if mode == "consensus":
        agreement = g.sign().mean(0).abs()
        # A pixel must have above-threshold sign agreement to contribute.
        if not 0 <= consensus_threshold < 1:
            raise ValueError("consensus_threshold must be in [0,1)")
        return g.mean(0) * (agreement > consensus_threshold)
    if mode not in {"equal", "weighted", "pcgrad", "weighted_pcgrad"}:
        raise ValueError(f"Unknown gradient fusion mode: {mode}")
    if weights is None:
        w = g.new_full((k, b), 1.0 / k)
    else:
        w = weights.float().to(g.device)
        if w.ndim == 1:
            w = w[:, None].expand(k, b)
        if w.shape != (k, b):
            raise ValueError("weights must have shape [K] or [K,B]")
        if torch.any(w < 0) or torch.any(~torch.isfinite(w)):
            raise ValueError("weights must be finite and nonnegative")
        w = w / w.sum(dim=0, keepdim=True).clamp_min(1e-12)
        if torch.any(w.sum(dim=0) <= 0):
            raise ValueError("At least one positive weight per image required")
    return (g * w[:, :, None, None, None]).sum(0)


def adaptive_weights_from_surrogate_lookahead(
    cross_surrogate_loss_improvement: Tensor,
    *,
    temperature: float = 0.20,
) -> Tensor:
    """Return [K,B] weights from cross-surrogate *loss decrease*, NEVER targets.

    Input [K,K,B]: for each trial direction i, improvement of surrogate j
    when applying a small projected one-step perturbation:
      delta_ij = L_j(current) - L_j(candidate_from_gradient_i).
    Exclude diagonal so the weights reward improvements on OTHER surrogates.
    All lookahead candidates must respect the same clean-image L_inf ball.
    This is AdaEA-inspired and NOT identical to AdaEA's classifier-logit score.
    """
    x = cross_surrogate_loss_improvement.float()
    if x.ndim != 3 or x.shape[0] != x.shape[1] or x.shape[0] < 2:
        raise ValueError("Expected [K,K,B] with K>=2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(x).all():
        raise ValueError("Surrogate loss improvements contain NaN/Inf")
    k = x.shape[0]
    off_diagonal = ~torch.eye(k, device=x.device, dtype=torch.bool)
    mean_cross_improvement = (x * off_diagonal[..., None]).sum(dim=1) / (k - 1)
    return torch.softmax(mean_cross_improvement / temperature, dim=0)


def differentiable_resize_pad(
    image: Tensor,
    *,
    min_scale: float = 0.80,
    max_scale: float = 1.00,
    generator: torch.Generator | None = None,
) -> Tensor:
    """DI-style differentiable resize+pad to original canvas.

    Use only as a stochastic surrogate forward transformation. Pixel
    perturbation projection is done on the ORIGINAL image in the attack runner,
    not after these augmented surrogate views. Generator/seed must be logged.
    """
    if image.ndim != 4:
        raise ValueError("Expected [B,C,H,W] input")
    if not (0 < min_scale <= max_scale <= 1):
        raise ValueError("Require 0 < min_scale <= max_scale <= 1")
    height, width = image.shape[-2:]
    # CPU sampling supports CPU-seeded reproducibility regardless of GPU.
    scale = float((torch.rand((), generator=generator) *
                   (max_scale - min_scale) + min_scale).item())
    nh = max(1, min(height, round(height * scale)))
    nw = max(1, min(width, round(width * scale)))
    resized = F.interpolate(
        image, size=(nh, nw), mode="bilinear", align_corners=False
    )
    top = int(torch.randint(height - nh + 1, (), generator=generator).item())
    left = int(torch.randint(width - nw + 1, (), generator=generator).item())
    return F.pad(resized, (left, width - nw - left, top, height - nh - top))
