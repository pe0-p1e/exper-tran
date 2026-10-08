import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from primary_ml_cka.models.common.loading import freeze_module


def load_processor(snapshot: Path):
    return AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def load_generative_proxy(
    snapshot: Path,
    device: torch.device,
    *,
    modules_to_not_convert: tuple[str, ...] = (),
    precision: str = "nf4",
):
    if device.type != "cuda":
        raise ValueError("Generative proxies require CUDA")
    if precision not in {"nf4", "bf16"}:
        raise ValueError(f"Unsupported generative proxy precision: {precision}")
    quantization_kwargs = {}
    if modules_to_not_convert:
        quantization_kwargs["llm_int8_skip_modules"] = list(modules_to_not_convert)
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": {"": device.index or 0},
        "torch_dtype": torch.bfloat16,
    }
    attention_implementation = os.environ.get("PRIMARY_ML_CKA_ATTN_IMPLEMENTATION")
    if attention_implementation:
        load_kwargs["attn_implementation"] = attention_implementation
    if precision == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            **quantization_kwargs,
        )
        load_kwargs["quantization_config"] = quantization
    elif modules_to_not_convert:
        raise ValueError("modules_to_not_convert is only supported with nf4 precision")
    model = AutoModelForImageTextToText.from_pretrained(snapshot, **load_kwargs)
    model.config.use_cache = False
    return freeze_module(model)
