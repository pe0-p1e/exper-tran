"""Differentiable surrogate-side V7 image-representation objectives.

No target model imports, target logits, or generated labels. Models may be BF16;
geometric calculations are FP32 for stability. This module is NOT wired into
the V7 launcher yet; use the focused unit tests before integrating.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def _unit(x: Tensor) -> Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-8)


def class_scores(
    image_vectors: Tensor,
    reference_features: Tensor,
    *,
    aggregation: Literal["centroid", "soft_reference"] = "centroid",
    reference_temperature: float = 0.05,
) -> Tensor:
    """Return [batch, class] surrogate cosine scores, not model class logits.

    image_vectors: [B,D] differentiable hidden-state pooled representations.
    reference_features: [C,R,D], 48 reference samples/class in the fixed protocol.
    Each class has its own surrogate features, hence feature dimensions need not
    agree between models. Never use the target model to create this bank.
    """
    if image_vectors.ndim != 2 or reference_features.ndim != 3:
        raise ValueError("Expected image [B,D] and reference [C,R,D]")
    if image_vectors.shape[-1] != reference_features.shape[-1]:
        raise ValueError("Image/reference embedding dimensions differ")
    if reference_features.shape[1] < 1:
        raise ValueError("Empty reference bank")
    z = _unit(image_vectors)
    refs = _unit(reference_features)
    if aggregation == "centroid":
        centers = _unit(refs.mean(dim=1))
        return z @ centers.T
    if aggregation == "soft_reference":
        if reference_temperature <= 0:
            raise ValueError("reference_temperature must be positive")
        sims = torch.einsum("bd,crd->bcr", z, refs)
        # Stable soft best-of-many class exemplars, with log(R) correction.
        return reference_temperature * (
            torch.logsumexp(sims / reference_temperature, dim=-1)
            - math.log(refs.shape[1])
        )
    raise ValueError(f"Unknown reference aggregation: {aggregation}")


def all_class_margin_loss(
    scores: Tensor,
    target_class: int,
    *,
    margin: float = 0.05,
    temperature: float = 0.05,
    negative_mode: Literal["hardest", "logsumexp"] = "logsumexp",
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """Minimize the target-vs-other-class *surrogate feature* margin loss.

    Labels are 1-based here, matching the V6 representation-source protocol.
    Unlike the previous source-only push, all 9 non-target classes contribute.
    """
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must be [B,C] with C>=2")
    if not 1 <= target_class <= scores.shape[1]:
        raise ValueError("Target class must be 1-based and within reference bank")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = scores.float()
    target = s[:, target_class - 1]
    negative = torch.cat((s[:, :target_class - 1], s[:, target_class:]), dim=1)
    if negative_mode == "hardest":
        rival = negative.max(dim=1).values
    elif negative_mode == "logsumexp":
        rival = temperature * torch.logsumexp(negative / temperature, dim=1)
    else:
        raise ValueError(f"Unknown negative_mode: {negative_mode}")
    per_image = F.softplus((rival - target + margin) / temperature) * temperature
    return per_image.mean() if reduction == "mean" else per_image


def original_pull_push_loss(
    image_vectors: Tensor,
    reference_features: Tensor,
    source_class: int,
    target_class: int,
    *,
    pull_weight: float = 0.75,
    push_weight: float = 0.25,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """V6 reference objective for an equivalent controlled comparison."""
    centers = _unit(_unit(reference_features).mean(dim=1))
    z = _unit(image_vectors)
    if source_class == target_class:
        raise ValueError("Source and target must differ")
    if not (1 <= source_class <= centers.shape[0] and
            1 <= target_class <= centers.shape[0]):
        raise ValueError("1-based class index out of bounds")
    if pull_weight < 0 or push_weight < 0:
        raise ValueError("Weights must be nonnegative")
    source_sim = (z * centers[source_class - 1]).sum(-1)
    target_sim = (z * centers[target_class - 1]).sum(-1)
    per_image = pull_weight * (1.0 - target_sim) + push_weight * (1.0 + source_sim)
    return per_image.mean() if reduction == "mean" else per_image


def local_patch_target_loss(
    adversarial_patch_tokens: Tensor,
    target_patch_centers: Tensor,
    source_patch_centers: Tensor,
    *,
    patch_mask: Tensor | None = None,
    temperature: float = 0.07,
    margin: float = 0.05,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """Patch-level contrastive term with *surrogate-specific* patch prototypes.

    Patch centers must be prepared from target/source reference images *of the
    same proxy, depth and preprocessing*. Suggested shape [K,D], K~8-32.
    Uses a smooth nearest-center score for each image patch; no cross-model
    token concatenation, no classifier outputs and no assumptions about a
    fixed patch grid. Exclude CLS/padding in the caller.
    """
    if adversarial_patch_tokens.ndim != 3:
        raise ValueError("adversarial_patch_tokens must be [B,P,D]")
    if target_patch_centers.ndim != 2 or source_patch_centers.ndim != 2:
        raise ValueError("patch centers must be [K,D]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    tokens = _unit(adversarial_patch_tokens)
    tar = _unit(target_patch_centers)
    src = _unit(source_patch_centers)
    if tokens.shape[-1] != tar.shape[-1] or tokens.shape[-1] != src.shape[-1]:
        raise ValueError("Patch feature dimensions mismatch")
    tgt_similarity = torch.matmul(tokens, tar.T)
    src_similarity = torch.matmul(tokens, src.T)
    target_softmax = temperature * (
        torch.logsumexp(tgt_similarity / temperature, dim=-1)
        - math.log(tar.shape[0])
    )
    source_softmax = temperature * (
        torch.logsumexp(src_similarity / temperature, dim=-1)
        - math.log(src.shape[0])
    )
    patch_loss = F.softplus(
        (source_softmax - target_softmax + margin) / temperature
    ) * temperature
    if patch_mask is not None:
        if patch_mask.shape != patch_loss.shape:
            raise ValueError("patch_mask shape must match [B,P]")
        weights = patch_mask.float()
        if (weights.sum(dim=1) <= 0).any():
            raise ValueError("Image with no valid patch tokens")
        per_image = (patch_loss * weights).sum(dim=1) / weights.sum(dim=1)
    else:
        per_image = patch_loss.mean(dim=1)
    return per_image.mean() if reduction == "mean" else per_image


def combine_multi_layer(
    losses: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> Tensor:
    """Sum independent *surrogate-side* layer losses with explicit weights.

    Prefer weights calibrated exclusively from held-out surrogate embeddings;
    avoid choosing weights on the real target's outcomes.
    """
    if not losses:
        raise ValueError("Need at least one selected layer")
    if weights is None:
        weights = {name: 1.0 for name in losses}
    if set(weights) != set(losses):
        raise ValueError("Layer and weight keys differ")
    if any(v < 0 for v in weights.values()):
        raise ValueError("Layer weights must be nonnegative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Layer weights sum to zero")
    return sum(losses[k] * (weights[k] / total) for k in losses)
