"""Black-box target generation backend.

This module is forbidden from proxy attack imports. It exposes decoded text only,
never target logits, hidden states, image representations, or gradients.
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

from primary_ml_cka.models.common.loading import freeze_module


def load_target_for_generation(snapshot: Path, device: torch.device, precision: str = "auto"):
    if device.type != "cuda":
        raise ValueError("Target generation requires CUDA")
    if precision not in {"auto", "bf16", "int8"}:
        raise ValueError("precision must be 'auto', 'bf16', or 'int8'")
    weight_bytes = sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))
    load_kwargs = {}
    if precision == "int8" or (precision == "auto" and weight_bytes > 12_000_000_000):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if os.environ.get("PRIMARY_ML_CKA_ATTN_IMPLEMENTATION"):
        load_kwargs["attn_implementation"] = os.environ["PRIMARY_ML_CKA_ATTN_IMPLEMENTATION"]
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index or 0},
        **load_kwargs,
    )
    model.config.use_cache = True
    return freeze_module(model)
