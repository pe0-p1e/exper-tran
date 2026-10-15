"""Detailed, machine-readable BF16 and quantization audit for recovery models."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from primary_ml_cka.models.common.precision import assert_bf16_unquantized, quantization_audit


def _save_records(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(path.read_text()) if path.is_file() else []
    # Keep an audit event for each load, not only the latest model/role tuple.
    record["load_sequence"] = len(records) + 1
    records.append(record)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def audit_loaded_model(
    model: torch.nn.Module,
    *,
    model_name: str,
    revision: str,
    role: str,
    repo_id: str,
    output_dir: Path,
    vision_depth: int | None = None,
) -> dict:
    """Audit immediately after model load; strict mismatches stop the caller."""
    assert_bf16_unquantized(model, model_name=f"{model_name} ({role})")
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    component_counts: dict[str, Counter[str]] = defaultdict(Counter)
    component_bytes: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[str]] = defaultdict(list)
    total_parameters = 0
    total_float_parameters = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        dtype = str(parameter.dtype).replace("torch.", "")
        device_counts[str(parameter.device)] += count
        if parameter.is_floating_point():
            total_float_parameters += count
            dtype_counts[dtype] += count
            dtype_bytes[dtype] += count * parameter.element_size()
            if len(samples[dtype]) < 20:
                samples[dtype].append(name)
            lower = name.lower()
            if any(token in lower for token in ("projector", "merger", "multi_modal_projector", "mm_projector")):
                component = "multimodal_projector"
            elif any(token in lower for token in ("vision", "visual", "vision_tower")):
                component = "vision_encoder"
            elif any(token in lower for token in ("language", "text_model", "text_decoder", "lm_head", "embed_tokens")):
                component = "language_backbone_or_head"
            else:
                component = "other"
            component_counts[component][dtype] += count
            component_bytes[component][dtype] += count * parameter.element_size()
    quant = quantization_audit(model)
    memory = {
        "cuda_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "cuda_reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0,
    }
    device_map = getattr(model, "hf_device_map", None)
    record = {
        "model_name": model_name,
        "repo_id": repo_id,
        "revision": revision,
        "role": role,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "total_parameter_count": total_parameters,
        "floating_parameter_count": total_float_parameters,
        "parameter_counts_by_dtype": dict(dtype_counts),
        "parameter_bytes_by_dtype": dict(dtype_bytes),
        "parameter_counts_by_device": dict(device_counts),
        "vision_encoder_parameter_counts_by_dtype": dict(component_counts["vision_encoder"]),
        "vision_encoder_parameter_bytes_by_dtype": dict(component_bytes["vision_encoder"]),
        "language_backbone_parameter_counts_by_dtype": dict(component_counts["language_backbone_or_head"]),
        "language_backbone_parameter_bytes_by_dtype": dict(component_bytes["language_backbone_or_head"]),
        "multimodal_projector_parameter_counts_by_dtype": dict(component_counts["multimodal_projector"]),
        "multimodal_projector_parameter_bytes_by_dtype": dict(component_bytes["multimodal_projector"]),
        "other_parameter_counts_by_dtype": dict(component_counts["other"]),
        "sample_parameter_names_by_dtype": dict(samples),
        "vision_depth": vision_depth,
        "quantization": quant,
        "device_placement": {"parameter_counts_by_device": dict(device_counts), "hf_device_map": device_map},
        "cuda_memory": memory,
        "all_floating_parameters_bf16": set(dtype_counts) <= {"bfloat16"},
        "status": "pass",
    }
    _save_records(output_dir / "audits/model_dtype_audit.json", record)
    return record
