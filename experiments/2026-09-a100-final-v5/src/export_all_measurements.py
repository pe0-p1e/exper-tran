#!/usr/bin/env python3
"""Snapshot every intended full-90 cell, including incomplete attack/evaluation work."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import load_experiment, transitions
from primary_ml_cka.domain.identifiers import get_pair

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
EXPERIMENT = Path(__file__).resolve().parents[1]
SHARE = OUTPUT / "shareable/priority3"
DEST = SHARE / "all_available_measurements"
PAIRS = ("P02", "P14", "P16", "P19", "P20", "P21", "P22", "P23")


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(prefix: str, data: dict) -> dict:
    return {
        f"{prefix}{key}": finite(value)
        for key, value in data.items()
        if isinstance(value, (int, float, bool)) and not isinstance(value, str)
    }


def main() -> None:
    raw = load_experiment(EXPERIMENT / "config/full90_selected_layer_main5.yaml")
    trans = transitions(raw)
    DEST.mkdir(parents=True, exist_ok=True)
    cell_rows, image_rows, coverage_rows, valid_states, stale_states = [], [], [], [], []
    errors = []
    for pair_id in PAIRS:
        pair = get_pair(pair_id)
        for transition in trans:
            tid = transition.transition_id
            state_path = OUTPUT / "states_layer_100" / pair_id / tid / "batch_00.json"
            state = None
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    errors.append({"state": str(state_path.relative_to(OUTPUT)), "error": str(exc)})
            status = state.get("status", "missing") if state else "missing"
            attack = state.get("attack", {}) if state else {}
            target = state.get("target", {}) if state else {}
            # A frozen attack remains valid when the proxy misses its target.
            attack_available = status in {"attack_complete", "complete"} and attack.get("status") in {"ok", "proxy_target_not_reached"}
            target_evaluated = status == "complete" and bool(target.get("target_hit_mask"))
            n = int(state.get("source_count", 0)) if state else 0
            if attack_available:
                valid_states.append(state_path)
            elif state is not None:
                stale_states.append(state_path)
            row = {
                "pair_id": pair_id,
                "proxy_model": pair.proxy_model,
                "target_model": pair.target_model,
                "family_relation": pair.exp_type.value,
                "transition_id": tid,
                "source_label": transition.source,
                "target_label": transition.target,
                "status": status,
                "attack_available": attack_available,
                "target_evaluated": target_evaluated,
                "N_attack": n if attack_available else None,
                "N_target": len(target["target_hit_mask"]) if target_evaluated else None,
                "state_json": str(state_path.relative_to(OUTPUT)) if state else None,
            }
            if state:
                for key in ("representation_layer", "lambda_cls", "lambda_cka", "source_logit_weight", "target_logit_weight", "step_size", "steps", "seed", "rho", "semantic_mode", "objective_tag"):
                    row[key] = state.get(key)
            if attack_available:
                row.update(scalar_metrics("attack_", attack))
                row["proxy_hits"] = attack.get("proxy_target_hit_count")
                denominator = attack.get("proxy_target_hit_denominator") or n
                row["PSR"] = row["proxy_hits"] / denominator if row["proxy_hits"] is not None and denominator else None
            if target_evaluated:
                row.update(scalar_metrics("target_", target.get("rates", {})))
                row["target_hits"] = sum(bool(x) for x in target["target_hit_mask"])
                row["unconditional_TASR"] = row["target_hits"] / len(target["target_hit_mask"])
                proxy_mask_for_rate = attack.get("proxy_target_hit_mask") or []
                target_mask_for_rate = target["target_hit_mask"]
                if len(proxy_mask_for_rate) != len(target_mask_for_rate):
                    raise ValueError(f"Proxy/target mask length mismatch: {pair_id}/{tid}")
                row["conditional_denominator"] = sum(bool(x) for x in proxy_mask_for_rate)
                row["target_hits_among_proxy_hits"] = sum(
                    bool(p) and bool(t)
                    for p, t in zip(proxy_mask_for_rate, target_mask_for_rate, strict=True)
                )
                row["TASR"] = (
                    row["target_hits_among_proxy_hits"] / row["conditional_denominator"]
                    if row["conditional_denominator"] else None
                )
            cell_rows.append(row)
            if not attack_available:
                continue
            ids = attack.get("source_image_ids") or []
            proxy_mask = attack.get("proxy_target_hit_mask") or []
            target_mask = target.get("target_hit_mask") or [] if target_evaluated else []
            clean_outputs = target.get("clean_outputs") or []
            adv_outputs = target.get("adversarial_outputs") or []
            image_dir = (
                OUTPUT / "attacks" / pair_id / f"a100_v5_{tid}" / "batch_00"
                / state["objective_tag"] / f"lambda_{float(state['lambda_cka']):g}"
            )
            for index in range(n):
                clean_png = image_dir / f"{index:02d}_clean.png"
                adv_png = image_dir / f"{index:02d}_adv.png"
                image_rows.append({
                    "pair_id": pair_id,
                    "transition_id": tid,
                    "image_index": index,
                    "source_label": transition.source,
                    "target_label": transition.target,
                    "source_image_id": ids[index] if index < len(ids) else None,
                    "clean_png": str(clean_png.relative_to(OUTPUT)),
                    "adversarial_png": str(adv_png.relative_to(OUTPUT)),
                    "clean_png_bytes": clean_png.stat().st_size if clean_png.is_file() else None,
                    "adversarial_png_bytes": adv_png.stat().st_size if adv_png.is_file() else None,
                    "proxy_target_hit": proxy_mask[index] if index < len(proxy_mask) else None,
                    "target_evaluated": target_evaluated,
                    "target_hit": target_mask[index] if index < len(target_mask) else None,
                    "clean_parsed_label": clean_outputs[index].get("parsed_label") if index < len(clean_outputs) else None,
                    "adversarial_parsed_label": adv_outputs[index].get("parsed_label") if index < len(adv_outputs) else None,
                    "clean_parser_status": clean_outputs[index].get("parser_status") if index < len(clean_outputs) else None,
                    "adversarial_parser_status": adv_outputs[index].get("parser_status") if index < len(adv_outputs) else None,
                })
    for pair_id in PAIRS:
        rows = [r for r in cell_rows if r["pair_id"] == pair_id]
        counts = Counter(r["status"] for r in rows)
        attack_rows = [r for r in rows if r["attack_available"]]
        target_rows = [r for r in rows if r["target_evaluated"]]
        coverage_rows.append({
            "pair_id": pair_id,
            "intended_cells": 90,
            "attack_complete_cells": len(attack_rows),
            "target_evaluated_cells": len(target_rows),
            "missing_cells": counts["missing"],
            "running_or_stale_cells": counts["running"],
            "attack_only_cells": counts["attack_complete"],
            "complete_cells": counts["complete"],
            "N_attack_images": sum(r["N_attack"] or 0 for r in attack_rows),
            "N_target_images": sum(r["N_target"] or 0 for r in target_rows),
            "proxy_success_denominator": sum(r.get("conditional_denominator", 0) for r in target_rows),
            "target_hits_among_proxy_hits": sum(r.get("target_hits_among_proxy_hits", 0) for r in target_rows),
            "partial_evaluated_TASR": (
                sum(r.get("target_hits_among_proxy_hits", 0) for r in target_rows)
                / sum(r.get("conditional_denominator", 0) for r in target_rows)
                if sum(r.get("conditional_denominator", 0) for r in target_rows) else None
            ),
        })
    write_csv(DEST / "full90_all_cells.csv", cell_rows)
    write_csv(DEST / "full90_all_images.csv", image_rows)
    missing_pngs = sum(row["clean_png_bytes"] is None or row["adversarial_png_bytes"] is None for row in image_rows)
    write_csv(DEST / "coverage.csv", coverage_rows)
    write_csv(DEST / "read_errors.csv", errors)
    metadata = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "intended_cells": len(cell_rows),
        "valid_attack_states": len(valid_states),
        "per_image_rows": len(image_rows),
        "missing_clean_or_adversarial_png_pairs": missing_pngs,
        "interpretation": "Partial target rates are descriptive for the evaluated subset only; evaluation order is not randomized. Missing target metrics are null, not zero. Stale running states are excluded.",
        "geometry": "Existing layer-pilot geometry and ablation summaries are included. New full-90 geometry is scoped to P02/P14/P23; deferred pairs retain all available attack, PNG, and evaluated-subset measurements without further geometry extraction.",
    }
    (DEST / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    report = [
        "# A100 V5 全部可用测量快照",
        "",
        f"生成时间：{metadata['snapshot_utc']}。共 {len(cell_rows)} 个计划 cell，{len(valid_states)} 个有效攻击状态，{len(image_rows)} 个逐图记录。",
        "",
        "`attack_complete` 已生成冻结 PNG，可测攻击与表示空间指标；`complete` 还具备目标预测标签。`running` 可能是停止后的残留状态，不纳入结果。空值表示尚未测量，不等于 0。主 TASR=proxy 命中且 target 命中/ proxy 命中；proxy 未命中的样本不计主分母。未完成 pair 的 target rate 只描述已评估子集，评估顺序并非随机，不能作为 full-90 TASR 估计。",
        "",
        "| Pair | 攻击完成/90 | 目标评估/90 | 攻击图像数 | 目标评估图像数 | proxy 命中分母 | 已评估子集条件 TASR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in coverage_rows:
        tasr = f"{r['partial_evaluated_TASR']:.1%}" if r["partial_evaluated_TASR"] is not None else "—"
        report.append(f"| {r['pair_id']} | {r['attack_complete_cells']} | {r['target_evaluated_cells']} | {r['N_attack_images']} | {r['N_target_images']} | {r['proxy_success_denominator']} | {tasr} |")
    report.extend([
        "", "## 文件说明", "",
        "`full90_all_cells.csv` 收录所有已有攻击标量，包括 proxy success、semantic gap、pull/push 相似度、L∞、耗时、显存，以及已有的目标 TASR。`full90_all_images.csv` 收录逐图 proxy/target 命中、预测标签和 PNG 路径。`raw_states/` 保留原始 JSON 和完整输出文本。`existing_geometry/` 包含已完成 layer pilot 的 CKA、RSA、representation shift、variance、margin、gap closure、相关矩阵等原空间测量；消融结果在 `ablation_summaries/`。",
        "", "## 指标口径", "",
        "主 TASR = proxy 与 target 同时命中目标类的图像数 / proxy 命中目标类的图像数。PSR = proxy 命中数 / clean-valid 图像数。`unconditional_TASR` 保留全 clean-valid 分母口径。分母为 0 时主 TASR 留空。聚合应先合计分子、分母，再相除。",
        "", "表示空间指标使用原始高维 embedding：CKA 是 proxy/target 在同组 reference 图上的线性 CKA；RSA 是图像相似度矩阵上三角的 Spearman 相关。Δpull = adversarial 对目标 prototype 的 cosine − clean cosine；Δpush = clean 对源 prototype 的 cosine − adversarial cosine；ΔR = Δpull + Δpush。gap closure 是 target−source cosine margin 的变化。类方差是参考云的平均 1−cosine，同时提供 covariance trace 和 effective rank。t-SNE 坐标距离仅作定性展示。",
        "", "`png_perturbations.csv` 测量保存后实际 PNG 的逐图 L∞、平均绝对变化、RMS 变化与修改像素比例；这些值从 RGB 字节重读计算，可用于核对 epsilon 预算与图像扰动强度。",
        "", "`attack_grad_component_cosine` 等与双梯度分量有关的字段，在 `lambda_cls=0` 且未启用 CKA 分量的主实验可能是记录用的 0 占位，不应解释成两个有效梯度的正交程度。原始 JSON 中的 NaN 在行级 CSV 中转为空值。",
        "", "P02/P14/P23 的 full-90 joint PCA/t-SNE 与表示空间测量仍在运行；五组暂缓组合保留已有 layer pilot geometry、攻击标量、逐图 PNG 指标和已评估 target 子集测量，不再启动 full-90 geometry。", "",
    ])
    (DEST / "README.md").write_text("\n".join(report), encoding="utf-8")
    archive_path = SHARE / "all_available_measurements.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(DEST.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
        for path in valid_states:
            archive.write(path, f"raw_states/{path.relative_to(OUTPUT / 'states_layer_100')}")
        for path in stale_states:
            archive.write(path, f"raw_stale_states_excluded/{path.relative_to(OUTPUT / 'states_layer_100')}")
        for path in sorted((OUTPUT / "analysis/layers").rglob("*.csv")):
            archive.write(path, f"existing_geometry/{path.relative_to(OUTPUT / 'analysis/layers')}")
        for path in sorted((SHARE / "completed_data_analysis").rglob("*")):
            if path.is_file():
                archive.write(path, f"completed_data_analysis/{path.relative_to(SHARE / 'completed_data_analysis')}")
        for path in sorted((SHARE / "concurrent_analysis").rglob("*")):
            if path.is_file():
                archive.write(path, f"concurrent_analysis/{path.relative_to(SHARE / 'concurrent_analysis')}")
        for path in sorted((OUTPUT / "analysis/single_proxy_full90_partial").rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".png"}:
                archive.write(path, f"partial_full90_geometry/{path.relative_to(OUTPUT / 'analysis/single_proxy_full90_partial')}")
        for path in sorted((OUTPUT / "summaries").glob("layer_*_results.csv")):
            archive.write(path, f"pilot_summaries/{path.name}")
        for path in sorted((OUTPUT / "summaries").glob("ablation_*.csv")):
            archive.write(path, f"ablation_summaries/{path.name}")
        for path in sorted((OUTPUT / "evaluation/manifests").rglob("*.jsonl")):
            archive.write(path, f"manifests/{path.relative_to(OUTPUT / 'evaluation/manifests')}")
        audit = OUTPUT / "analysis/model_layer_audit.csv"
        if audit.is_file():
            archive.write(audit, "model_layer_audit.csv")
        for path in sorted((EXPERIMENT / "config").glob("*.yaml")):
            archive.write(path, f"configs/{path.name}")
        archive.write(Path(__file__), "code/export_all_measurements.py")
        archive.write(EXPERIMENT / "src/summarize_partial_geometry.py", "code/summarize_partial_geometry.py")
        archive.write(EXPERIMENT / "src/analyze_during_evaluation.py", "code/analyze_during_evaluation.py")
        archive.write(EXPERIMENT / "src/measure_png_perturbations.py", "code/measure_png_perturbations.py")
    print(json.dumps({"archive": str(archive_path), "bytes": archive_path.stat().st_size, "coverage": coverage_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
