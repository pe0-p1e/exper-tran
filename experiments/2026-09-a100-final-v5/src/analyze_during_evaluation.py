#!/usr/bin/env python3
"""CPU-only, coverage-aware analysis while target evaluation is still running."""

from __future__ import annotations

import filecmp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
MEASURE = OUTPUT / "shareable/priority3/all_available_measurements"
DEST = OUTPUT / "shareable/priority3/concurrent_analysis"
PRIORITY = ("P02", "P14", "P23")
ALL_PAIRS = ("P02", "P14", "P16", "P19", "P20", "P21", "P22", "P23")


def rho(frame: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    observed = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(observed) < 3 or observed[left].nunique() < 2 or observed[right].nunique() < 2:
        return np.nan, len(observed)
    return float(spearmanr(observed[left], observed[right]).statistic), len(observed)


def is_true(value) -> bool:
    return str(value).lower() == "true"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(MEASURE / "full90_all_cells.csv")
    images = pd.read_csv(MEASURE / "full90_all_images.csv", low_memory=False)
    frozen = cells[cells.attack_available].copy()
    diagnostics = []
    for pair, group in frozen.groupby("pair_id"):
        diagnostics.append({
            "pair_id": pair,
            "frozen_cells": len(group),
            "frozen_images": int(group.N_attack.sum()),
            "proxy_hits": int(group.proxy_hits.sum()),
            "PSR": group.proxy_hits.sum() / group.N_attack.sum(),
            "mean_semantic_gap_gain": group.attack_semantic_gap_gain.mean(),
            "median_semantic_gap_gain": group.attack_semantic_gap_gain.median(),
            "mean_attack_seconds_per_cell": group.attack_elapsed_seconds.mean(),
            "sum_attack_hours": group.attack_elapsed_seconds.sum() / 3600,
            "max_peak_reserved_vram_gb": group.attack_peak_reserved_vram_gb.max(),
            "max_linf_png": group.attack_linf_png.max(),
        })
    pd.DataFrame(diagnostics).to_csv(DEST / "proxy_diagnostics_all_frozen.csv", index=False)

    evaluated_all = cells[cells.target_evaluated & cells.pair_id.isin(ALL_PAIRS)].copy()
    partial_rates = []
    for pair in ALL_PAIRS:
        group = evaluated_all[evaluated_all.pair_id == pair]
        if group.empty:
            continue
        denominator = int(group.conditional_denominator.sum())
        hits = int(group.target_hits_among_proxy_hits.sum())
        clean_n = int(group.N_target.sum())
        partial_rates.append({
            "pair_id": pair,
            "evaluated_directions": len(group),
            "intended_directions": 90,
            "clean_valid_N": clean_n,
            "proxy_success_denominator": denominator,
            "target_hits_among_proxy_success": hits,
            "conditional_TASR": hits / denominator if denominator else np.nan,
            "unconditional_target_rate": group.target_hits.sum() / clean_n if clean_n else np.nan,
            "proxy_success_rate": group.proxy_hits.sum() / clean_n if clean_n else np.nan,
        })
    pd.DataFrame(partial_rates).to_csv(DEST / "partial_target_rates_all_pairs.csv", index=False)

    evaluated = evaluated_all[evaluated_all.pair_id.isin(PRIORITY)].copy()
    common_ids = set.intersection(*(
        set(evaluated[evaluated.pair_id == pair].transition_id) for pair in PRIORITY
    ))
    common = evaluated[evaluated.transition_id.isin(common_ids)]
    common_rows = []
    for pair in PRIORITY:
        group = common[common.pair_id == pair]
        denominator = int(group.conditional_denominator.sum())
        numerator = int(group.target_hits_among_proxy_hits.sum())
        common_rows.append({
            "pair_id": pair,
            "common_evaluated_directions": len(group),
            "N_clean_valid": int(group.N_target.sum()),
            "N_proxy_success": denominator,
            "N_target_success_among_proxy_success": numerator,
            "conditional_TASR": numerator / denominator if denominator else np.nan,
            "unconditional_TASR": group.target_hits.sum() / group.N_target.sum() if len(group) else np.nan,
            "PSR": group.proxy_hits.sum() / group.N_target.sum() if len(group) else np.nan,
        })
    pd.DataFrame(common_rows).to_csv(DEST / "priority3_common_direction_rates.csv", index=False)

    # P14 and P23 share the same Qwen2B proxy and fixed attack settings. Verify
    # actual bytes rather than assuming their adversarial artifacts are paired.
    p14 = images[images.pair_id == "P14"].set_index(["transition_id", "image_index"])
    p23 = images[images.pair_id == "P23"].set_index(["transition_id", "image_index"])
    if not p14.index.equals(p23.index):
        raise RuntimeError("P14 and P23 do not cover the same image indices")
    identity_rows = []
    paired_rows = []
    for (tid, index), left in p14.iterrows():
        right = p23.loc[(tid, index)]
        if left.source_image_id != right.source_image_id:
            raise RuntimeError(f"Canonical clean image differs: {tid}/{index}")
        left_png = OUTPUT / left.adversarial_png
        right_png = OUTPUT / right.adversarial_png
        equal = filecmp.cmp(left_png, right_png, shallow=False)
        identity_rows.append({"transition_id": tid, "image_index": index, "adversarial_png_identical": equal})
        if tid in common_ids and equal and is_true(left.target_evaluated) and is_true(right.target_evaluated):
            if is_true(left.proxy_target_hit) and is_true(right.proxy_target_hit):
                paired_rows.append({
                    "transition_id": tid,
                    "image_index": index,
                    "source_image_id": left.source_image_id,
                    "target_label": left.target_label,
                    "P14_target_hit": is_true(left.target_hit),
                    "P23_target_hit": is_true(right.target_hit),
                    "adversarial_png_identical": equal,
                })
    identity = pd.DataFrame(identity_rows)
    identity.to_csv(DEST / "P14_P23_png_identity.csv", index=False)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(DEST / "P14_P23_paired_target_outcomes.csv", index=False)

    correlations = []
    for pair in ALL_PAIRS:
        group = evaluated_all[evaluated_all.pair_id == pair]
        if group.empty:
            continue
        for metric in ("PSR", "attack_semantic_gap_gain", "attack_elapsed_seconds", "attack_peak_reserved_vram_gb"):
            value, n = rho(group, "TASR", metric)
            correlations.append({"pair_id": pair, "metric": metric, "spearman_rho_with_conditional_TASR": value, "n_evaluated_cells": n})
    pd.DataFrame(correlations).to_csv(DEST / "partial_target_correlations.csv", index=False)

    asymmetry_rows = []
    for pair in ALL_PAIRS:
        group = evaluated_all[evaluated_all.pair_id == pair]
        directed = {(int(row.source_label), int(row.target_label)): row for row in group.itertuples()}
        for (source, target), forward in directed.items():
            reverse = directed.get((target, source))
            if source >= target or reverse is None:
                continue
            if not np.isfinite(forward.TASR) or not np.isfinite(reverse.TASR):
                continue
            asymmetry_rows.append({
                "pair_id": pair,
                "source_label": source,
                "target_label": target,
                "forward_transition_id": forward.transition_id,
                "reverse_transition_id": reverse.transition_id,
                "forward_conditional_TASR": forward.TASR,
                "reverse_conditional_TASR": reverse.TASR,
                "delta_TASR": forward.TASR - reverse.TASR,
                "forward_proxy_success_N": forward.conditional_denominator,
                "reverse_proxy_success_N": reverse.conditional_denominator,
            })
    pd.DataFrame(asymmetry_rows).to_csv(DEST / "partial_target_asymmetry.csv", index=False)

    pair_rates = {row["pair_id"]: row for row in common_rows}
    pair_count = int(identity.adversarial_png_identical.sum())
    both = int((paired.P14_target_hit & paired.P23_target_hit).sum()) if len(paired) else 0
    p14_only = int((paired.P14_target_hit & ~paired.P23_target_hit).sum()) if len(paired) else 0
    p23_only = int((~paired.P14_target_hit & paired.P23_target_hit).sum()) if len(paired) else 0
    neither = int((~paired.P14_target_hit & ~paired.P23_target_hit).sum()) if len(paired) else 0
    lines = [
        "# 目标评估进行中的同步 CPU 分析", "",
        f"当前三组共同已评估方向数：**{len(common_ids)}/90**。下表只比较这批相同方向；它不是最终 full-90 结果。每个 TASR 的分母仅含 proxy 命中的图像。", "",
        "| Pair | 共同方向 | proxy 命中分母 | target 条件命中 | 条件 TASR | PSR |", "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in PRIORITY:
        item = pair_rates[pair]
        tasr = f"{item['conditional_TASR']:.2%}" if np.isfinite(item["conditional_TASR"]) else "—"
        lines.append(f"| {pair} | {item['common_evaluated_directions']} | {item['N_proxy_success']} | {item['N_target_success_among_proxy_success']} | {tasr} | {item['PSR']:.2%} |")
    lines.extend([
        "", f"P14 与 P23 的冻结 adversarial PNG 共比较 **{len(identity)}** 张，字节完全一致 **{pair_count}** 张。只对共同已评估、PNG 完全一致且两边 proxy 都命中的 {len(paired)} 张做严格逐图配对：两者都命中 {both}，仅 P14 命中 {p14_only}，仅 P23 命中 {p23_only}，两者都未命中 {neither}。", "",
        "两组虽共享 Qwen2B proxy 和配置，但部分 adversarial PNG 字节不同，不能把全部样本当成完全相同的攻击输入。严格配对子集控制了 clean 图、攻击预算与实际 adversarial 像素；目标模型仍不同，结果不能单独归因于家族差异。", "",
        "`proxy_diagnostics_all_frozen.csv` 覆盖所有已冻结攻击 cell 的 PSR、semantic gap、耗时、显存与 L∞；`partial_target_correlations.csv` 覆盖已有 target 评估的 pair-cell 计算探索性 Spearman。其评估顺序并非随机，完成 90 向后须重算。", "",
        "五组暂缓组合的现有 target 评估子集和条件率见 `partial_target_rates_all_pairs.csv`；对应双向都已测且条件分母为正的方向差见 `partial_target_asymmetry.csv`。评估不是随机抽样，所以这些率仅描述已完成部分。", "",
        f"目前全部 pair 共得到 {len(asymmetry_rows)} 个双向均已评估且两边条件分母都大于 0 的无序类别对；这些不对称率同样只用于描述已测子集。", "",
    ])
    (DEST / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"common_directions={len(common_ids)} identical_pngs={pair_count}/{len(identity)} paired_proxy_hits={len(paired)}")


if __name__ == "__main__":
    main()
