"""DINOv2 visual-only proxy with differentiable V5-style spatial pooling."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.types import ImageEmbeddingOutput, TapContract
from primary_ml_cka.models.proxies.base import BaseProxy
from primary_ml_cka.models.representations import RepresentationSpec, resolve_vision_layer
from primary_ml_cka.models.taps.pooling import masked_mean_l2


class DINOv2Proxy(BaseProxy):
    def __init__(self, model: torch.nn.Module, *, image_size: int = 518) -> None:
        self.model = model
        self.model_id = "facebook/dinov2-large"
        self.image_size = int(image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=next(model.parameters()).device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=next(model.parameters()).device).view(1, 3, 1, 1)

    def image_embeddings(self, images: torch.Tensor, *, representation_type: str = "vision_encoder", layer: int = -1, pooling: str = "mean") -> ImageEmbeddingOutput[torch.Tensor]:
        spec = RepresentationSpec(representation_type, layer, pooling)
        spec.validate()
        if representation_type != "vision_encoder":
            raise ValueError("DINOv2 exposes the vision_encoder representation only")
        total = int(self.model.config.num_hidden_layers)
        resolved = resolve_vision_layer(layer, total)
        pixels = functional.interpolate(images.float(), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False, antialias=True)
        pixels = ((pixels - self.mean) / self.std).to(dtype=next(self.model.parameters()).dtype)
        output = self.model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        # DINOv2 token zero is CLS. V5 spatial-token mean semantics use patch tokens.
        tokens = output.hidden_states[resolved + 1][:, 1:, :]
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        pooled = masked_mean_l2(tokens, mask)
        tap = TapContract(
            self.model_id,
            MODEL_REVISIONS[self.model_id],
            f"encoder.layer.{resolved}",
            "DINOv2 selected vision block; CLS token excluded",
            "masked mean over spatial patch tokens",
            "per-image L2 after FP32 mean pooling",
            str(tokens.dtype).removeprefix("torch."),
            "validated_by_forward",
            tuple(tokens.shape),
            "spatial patch tokens only",
            representation_type=representation_type,
            requested_layer=layer,
            resolved_layer=resolved,
            total_vision_layers=total,
        )
        return ImageEmbeddingOutput(pooled if pooling == "mean" else tokens, tokens, mask, tap, semantic_embeddings=pooled if pooling == "mean" else None)

    def target_loss(self, *args, **kwargs):
        raise RuntimeError("DINOv2 is visual-only; V6 scores it against cached class prototypes")

    def free_generate_labels(self, *args, **kwargs):
        raise RuntimeError("DINOv2 has no language generation head")
