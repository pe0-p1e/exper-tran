"""Strict BF16 parameter checks used by the A100 recovery campaign."""

from __future__ import annotations

import os

import torch


class ModelPrecisionError(RuntimeError):
    pass


def quantization_audit(model: torch.nn.Module) -> dict:
    quantized_modules = []
    for name, module in model.named_modules():
        cls = f"{type(module).__module__}.{type(module).__name__}".lower()
        if any(token in cls for token in ("4bit", "8bit", "quantlinear", "gptq", "awq", "quanto", "hqq", "fp8")):
            quantized_modules.append({"name": name, "class": f"{type(module).__module__}.{type(module).__name__}"})
    config = getattr(model, "config", None)
    quant_config = getattr(config, "quantization_config", None) if config is not None else None
    return {
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "is_quantized": bool(getattr(model, "is_quantized", False)),
        "quantization_method": str(getattr(model, "quantization_method", None)),
        "config_quantization": str(quant_config) if quant_config is not None else None,
        "quantized_modules": quantized_modules,
    }


def assert_bf16_unquantized(model: torch.nn.Module, *, model_name: str = "model") -> None:
    """Hard-fail before inference if any floating parameter is not BF16."""
    quant = quantization_audit(model)
    if (
        quant["is_loaded_in_4bit"]
        or quant["is_loaded_in_8bit"]
        or quant["is_quantized"]
        or quant["quantized_modules"]
        or quant["config_quantization"] is not None
    ):
        raise ModelPrecisionError(f"{model_name} is quantized: {quant}")
    bad = [(name, str(param.dtype)) for name, param in model.named_parameters() if param.is_floating_point() and param.dtype != torch.bfloat16]
    if bad:
        raise ModelPrecisionError(f"{model_name} has {len(bad)} non-BF16 floating parameters; examples={bad[:12]}")


def strict_bf16_enabled() -> bool:
    return os.environ.get("PRIMARY_ML_CKA_STRICT_BF16") == "1"
