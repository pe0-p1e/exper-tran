#!/usr/bin/env python3
"""Summarize every completed layer pilot and ablation without using the GPU."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
DEST = OUTPUT / "shareable/priority3/completed_data_analysis"
PAIRS = ("P02", "P14", "P23", "P16", "P19", "P20", "P21", "P22")
DEPTHS = (1, 25, 50, 75, 100)
CROSS = {"P02", "P23"}
COLORS = {"cross": "#cf702d", "intra": "#315b87"}


def markdown_table(frame: pd.DataFrame, *, decimals: int = 1) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    rows.append("|" + "|".join("---" for _ in frame.columns) + "|")
    for _, row in frame.iterrows():
        values = [
            "—" if pd.isna(value) else (f"{value:.{decimals}f}" if isinstance(value, float) else str(value))
            for value in row
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    summary_dir = OUTPUT / "summaries"
    key_set = None
    layer_rows = []
    geometry_rows = []
    for depth in DEPTHS:
        summary = pd.read_csv(summary_dir / f"layer_{depth:03d}_results.csv")
        keys = set(zip(summary.pair_id, summary.transition_id, strict=True))
        if len(summary) != 80 or len(keys) != 80:
            raise RuntimeError(f"Layer {depth} must have 8 pairs × 10 unique transitions")
        if key_set is not None and keys != key_set:
            raise RuntimeError("Layer summaries use different pair/transition cells")
        key_set = keys
        if not (summary.images == 50).all():
            raise RuntimeError("Pilot cell denominator changed")
        for pair in PAIRS:
            cells = summary[summary.pair_id == pair]
            geo = pd.read_csv(
                OUTPUT / "analysis/layers" / f"layer_{depth:03d}" / pair
                / "representation_shift.csv"
            )
            if len(cells) != 10 or len(geo) != 10:
                raise RuntimeError(f"Missing pilot cells/geometry for {pair}, depth={depth}")
            layer_rows.append(
                {
                    "pair_id": pair,
                    "family_relation": "cross" if pair in CROSS else "intra",
                    "layer_percent": depth,
                    "cells": 10,
                    "N": int(cells.images.sum()),
                    "tasr_hits": int(cells.eligible_tasr_hits.sum()),
                    "tasr_denominator": int(cells.eligible_tasr_denominator.sum()),
                    "tasr_percent": 100 * cells.eligible_tasr_hits.sum() / cells.eligible_tasr_denominator.sum() if cells.eligible_tasr_denominator.sum() else np.nan,
                    "unconditional_tasr_percent": 100 * cells.tasr_hits.sum() / cells.images.sum(),
                    "proxy_hits": int(cells.proxy_hits.sum()),
                    "psr_percent": 100 * cells.proxy_hits.sum() / cells.images.sum(),
                    "asr_hits": int(cells.asr_hits.sum()),
                    "asr_percent": 100 * cells.asr_hits.sum() / cells.images.sum(),
                    "mean_cka": geo.CKA.mean(),
                    "mean_rsa": geo.RSA.mean(),
                    "mean_delta_R": geo.mean_delta_R.mean(),
                    "mean_clean_target_margin": geo.clean_target_margin.mean(),
                }
            )
            geo.insert(0, "layer_percent", depth)
            geo.insert(0, "family_relation", "cross" if pair in CROSS else "intra")
            geometry_rows.append(geo)
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(DEST / "layer_by_pair.csv", index=False)
    all_geometry = pd.concat(geometry_rows, ignore_index=True)
    all_geometry.to_csv(DEST / "pilot_geometry_all_cells.csv", index=False)

    ablation_rows = []
    ablation_keys = None
    for arm, label in (("pull_only", "Pull only"), ("push_only", "Push only"), ("pull_push", "Pull + push")):
        frame = pd.read_csv(summary_dir / f"ablation_{arm}.csv")
        keys = set(zip(frame.pair_id, frame.transition_id, strict=True))
        if len(frame) != 30 or len(keys) != 30 or not (frame.images == 50).all():
            raise RuntimeError(f"Ablation {arm} is incomplete")
        if ablation_keys is not None and keys != ablation_keys:
            raise RuntimeError("Ablation arms use different pair/transition cells")
        ablation_keys = keys
        for pair, cells in frame.groupby("pair_id"):
            ablation_rows.append(
                {
                    "pair_id": pair,
                    "arm": arm,
                    "label": label,
                    "N": int(cells.images.sum()),
                    "tasr_hits": int(cells.eligible_tasr_hits.sum()),
                    "tasr_denominator": int(cells.eligible_tasr_denominator.sum()),
                    "tasr_percent": 100 * cells.eligible_tasr_hits.sum() / cells.eligible_tasr_denominator.sum() if cells.eligible_tasr_denominator.sum() else np.nan,
                    "unconditional_tasr_percent": 100 * cells.tasr_hits.sum() / cells.images.sum(),
                    "asr_percent": 100 * cells.asr_hits.sum() / cells.images.sum(),
                }
            )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(DEST / "ablation_by_pair.csv", index=False)

    asymmetry_rows = []
    class_variance_rows = []
    for pair in PAIRS:
        directory = OUTPUT / "analysis/layers/layer_100" / pair
        asymmetry = pd.read_csv(directory / "asymmetry.csv")
        if len(asymmetry) != 10:
            raise RuntimeError(f"Layer-100 asymmetry missing for {pair}")
        asymmetry = asymmetry[asymmetry.source_label < asymmetry.target_label].copy()
        layer100 = pd.read_csv(summary_dir / "layer_100_results.csv")
        conditional = {
            (int(row.source), int(row.target)): (
                row.eligible_tasr_hits / row.eligible_tasr_denominator
                if row.eligible_tasr_denominator else np.nan
            )
            for row in layer100[layer100.pair_id == pair].itertuples()
        }
        asymmetry["TASR_forward"] = [conditional[(int(row.source_label), int(row.target_label))] for row in asymmetry.itertuples()]
        asymmetry["TASR_reverse"] = [conditional[(int(row.target_label), int(row.source_label))] for row in asymmetry.itertuples()]
        asymmetry["delta_TASR"] = asymmetry.TASR_forward - asymmetry.TASR_reverse
        asymmetry.insert(0, "pair_id", pair)
        asymmetry.insert(1, "family_relation", "cross" if pair in CROSS else "intra")
        asymmetry_rows.append(asymmetry)
        variance = pd.read_csv(directory / "class_variance.csv")
        if len(variance) != 10:
            raise RuntimeError(f"Layer-100 variance missing for {pair}")
        class_variance_rows.append(variance)
    asymmetry = pd.concat(asymmetry_rows, ignore_index=True)
    asymmetry.to_csv(DEST / "pilot_asymmetry_unique_pairs.csv", index=False)
    variance = pd.concat(class_variance_rows, ignore_index=True)
    duplicate_spread = variance.groupby(["target_model", "label"]).dispersion.agg(
        lambda values: values.max() - values.min()
    )
    if (duplicate_spread > 1e-5).any():
        raise RuntimeError("Shared target-model class variance changed across pair IDs")
    variance = variance.drop_duplicates(["target_model", "label"]).copy()
    variance.to_csv(DEST / "class_variance_by_target_model.csv", index=False)

    # Figure 1: same denominator and palette for target and proxy rates.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("tasr_percent", "psr_percent"),
        ("Target transfer success (TASR)", "Proxy targeted success (PSR)"),
        strict=True,
    ):
        matrix = layers.pivot(index="pair_id", columns="layer_percent", values=metric).loc[PAIRS, DEPTHS]
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(DEPTHS)), [f"{depth}%" for depth in DEPTHS])
        ax.set_yticks(range(len(PAIRS)), PAIRS)
        ax.set_xlabel("Normalized vision-layer depth")
        for i in range(len(PAIRS)):
            for j in range(len(DEPTHS)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 55 else "#1f2933")
    fig.colorbar(image, ax=axes, label="TASR: proxy hits; PSR: clean-valid", shrink=0.84)
    fig.savefig(DEST / "layer_tasr_psr_heatmap.png", dpi=180)
    plt.close(fig)

    # Figure 2: all 40 pair-layer observations, with shape and color indicating family.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, x, label in zip(
        axes,
        ("mean_cka", "mean_delta_R"),
        ("Mean CKA on class references", "Mean victim representation shift ΔR"),
        strict=True,
    ):
        for relation, marker in (("cross", "o"), ("intra", "s")):
            subset = layers[layers.family_relation == relation]
            ax.scatter(subset[x], subset.tasr_percent, s=42, alpha=0.75,
                       c=COLORS[relation], marker=marker, label=relation)
        ax.set_xlabel(label)
        ax.set_ylabel("Target TASR (%)")
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.16)
    axes[0].legend(frameon=False)
    fig.savefig(DEST / "pilot_geometry_vs_transfer.png", dpi=180)
    plt.close(fig)

    # Figure 3: method ablation on the same three pairs and matched 50-image cells.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(3)
    arms = (("pull_only", "#8aa7c2"), ("push_only", "#d8aa5e"), ("pull_push", "#315b87"))
    for index, (arm, color) in enumerate(arms):
        values = [
            float(ablation[(ablation.pair_id == pair) & (ablation.arm == arm)].tasr_percent.iloc[0])
            for pair in ("P02", "P14", "P23")
        ]
        offsets = x + (index - 1) * 0.25
        ax.bar(offsets, values, width=0.23, color=color, label=dict(
            pull_only="Pull only", push_only="Push only", pull_push="Pull + push"
        )[arm])
        for xx, value in zip(offsets, values, strict=True):
            ax.text(xx, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, ("P02 cross", "P14 intra", "P23 cross"))
    ax.set_ylabel("Target TASR (%)")
    ax.set_ylim(0, max(ablation.tasr_percent) * 1.16)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(DEST / "ablation_by_pair.png", dpi=180)
    plt.close(fig)

    # Figure 4: exploratory direction-level factors (five unordered class pairs per model pair).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, feature, label in zip(
        axes,
        ("variance_difference", "gap_closure_difference"),
        ("Source − target class dispersion", "Forward − reverse gap closure"),
        strict=True,
    ):
        for relation, marker in (("cross", "o"), ("intra", "s")):
            subset = asymmetry[asymmetry.family_relation == relation]
            ax.scatter(subset[feature], 100 * subset.delta_TASR, s=40, alpha=0.75,
                       c=COLORS[relation], marker=marker, label=relation)
        ax.axhline(0, color="#5b6570", linewidth=0.8)
        ax.axvline(0, color="#5b6570", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel("TASR forward − reverse (pp)")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    fig.savefig(DEST / "pilot_asymmetry_factors.png", dpi=180)
    plt.close(fig)

    # Figure 5: cosine dispersion is defined in each target encoder's own space.
    target_names = {
        "Qwen/Qwen3.5-2B": "Qwen 2B", "Qwen/Qwen3.5-4B": "Qwen 4B",
        "OpenGVLab/InternVL3_5-2B-HF": "InternVL 2B",
        "OpenGVLab/InternVL3_5-4B-HF": "InternVL 4B",
        "google/gemma-4-E2B-it": "Gemma E2B", "google/gemma-4-E4B-it": "Gemma E4B",
    }
    variance_matrix = variance.pivot(index="target_model", columns="label", values="dispersion")
    ordered_targets = [model for model in target_names if model in variance_matrix.index]
    variance_matrix = variance_matrix.loc[ordered_targets, range(1, 11)]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    image = ax.imshow(variance_matrix, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(10), [str(i) for i in range(1, 11)])
    ax.set_yticks(range(len(ordered_targets)), [target_names[name] for name in ordered_targets])
    ax.set_xlabel("ImageNet diverse-10 class ID")
    ax.set_title("Reference-class cosine dispersion by target encoder")
    fig.colorbar(image, ax=ax, label="Mean 1 − cosine to class center", shrink=0.85)
    fig.savefig(DEST / "class_variance_by_target.png", dpi=180)
    plt.close(fig)

    rate_matrix = layers.pivot(index="pair_id", columns="layer_percent", values="tasr_percent").loc[PAIRS, DEPTHS]
    priority = rate_matrix.loc[["P02", "P14", "P23"]]
    overall_priority = priority.mean(axis=0)
    max_pair_layer = rate_matrix.idxmax(axis=1)
    main3 = layers[(layers.layer_percent == 100) & layers.pair_id.isin(("P02", "P14", "P23"))]
    cka_rho = spearmanr(layers.mean_cka, layers.tasr_percent, nan_policy="omit").statistic
    shift_rho = spearmanr(layers.mean_delta_R, layers.tasr_percent, nan_policy="omit").statistic
    signed_var_rho = spearmanr(asymmetry.variance_difference, asymmetry.delta_TASR, nan_policy="omit").statistic
    signed_gap_rho = spearmanr(asymmetry.gap_closure_difference, asymmetry.delta_TASR, nan_policy="omit").statistic
    inventory = []
    for pair in PAIRS:
        states = list((OUTPUT / "states_layer_100" / pair).glob("T*/batch_00.json"))
        counts = Counter(json.loads(path.read_text()).get("status") for path in states)
        inventory.append(
            {
                "Pair": pair,
                "Attack PNG complete": int(counts["attack_complete"] + counts["complete"]),
                "Target evaluated": int(counts["complete"]),
                "Expected full90": 90,
            }
        )
    pd.DataFrame(inventory).to_csv(DEST / "experiment_inventory_snapshot.csv", index=False)
    priority_complete = all(
        item["Target evaluated"] == 90
        for item in inventory
        if item["Pair"] in {"P02", "P14", "P23"}
    )
    scope_sentence = (
        "P02/P14/P23 的 full-90 target 评估已完成；本报告聚焦原始 8-pair 层实验与三组方法消融，完整主结果见优先三 pair 的总报告。"
        if priority_complete
        else "full-90 target 评估正在跑 P02/P14/P23；以下主张不把 P16/P19/P20/P21/P22 的部分 full-90 状态当作完整结果。"
    )

    pair_table = pd.DataFrame(
        [
            {"Pair": pair, **{f"{depth}%": rate_matrix.loc[pair, depth] for depth in DEPTHS},
             "Peak layer": f"{max_pair_layer.loc[pair]}%"}
            for pair in PAIRS
        ]
    )
    ablation_table = ablation.pivot(index="pair_id", columns="arm", values="tasr_percent").loc[["P02", "P14", "P23"]]
    ablation_table = ablation_table.rename(columns={
        "pull_only": "Pull only", "push_only": "Push only", "pull_push": "Pull + push"
    }).reset_index().rename(columns={"pair_id": "Pair"})
    gap_table = main3[["pair_id", "psr_percent", "tasr_percent", "mean_cka", "mean_delta_R"]].copy()
    gap_table.columns = ["Pair", "PSR %", "TASR %", "Mean CKA", "Mean ΔR"]
    gap_table = gap_table.sort_values("Pair")

    pooled = {
        depth: (100 * group.tasr_hits.sum() / group.tasr_denominator.sum()) if group.tasr_denominator.sum() else np.nan
        for depth, group in layers.groupby("layer_percent")
    }
    pair_depth_rate = {
        (row.pair_id, row.layer_percent): row.tasr_percent
        for row in layers.itertuples()
    }
    combined_ablation = {
        arm: (100 * group.tasr_hits.sum() / group.tasr_denominator.sum()) if group.tasr_denominator.sum() else np.nan
        for arm, group in ablation.groupby("arm")
    }
    push_only_rate = (
        f"{combined_ablation['push_only']:.2f}%"
        if np.isfinite(combined_ablation["push_only"]) else "未定义（proxy 命中 0 张）"
    )

    report = f"""# 已完成实验数据的综合分析（A100 v5）

此报告汇总**全部已完整结束**的 layer pilot（8 pair × 5 layers × 10 directions × 50 images）与 pull/push 消融（3 pair × 3 arms × 10 directions × 50 images）。{scope_sentence}

## 数据覆盖与可比性

每个 layer pilot CSV 有 80 个唯一 pair-transition cell、每 cell 50 张 clean-valid 图，5 层使用同一组 80 个 cell。三组 ablation 各有 30 个相同的 pair-transition cell，每组 1,500 张。主 TASR 的分母是 proxy 命中数；PSR/ASR 的分母是所有 clean-valid 图。两种 TASR 口径均保存在行级 CSV。**proxy 命中数为 0 时条件 TASR 未定义，不写作 0%。**下面是运行时的 full-90 状态快照（会随评估推进）：

{markdown_table(pd.DataFrame(inventory), decimals=0)}

## 关键发现 1：最终层不是每个 pair 的最优层

表中是 pilot TASR（%，每 pair 每层 N=500）。

{markdown_table(pair_table)}

原 full-90 按 clean-valid 口径选择了 100% 层。按当前 proxy 条件分母重新计算，全部 8 pair 合并时，100% 层为 {pooled[100]:.2f}%，75% 为 {pooled[75]:.2f}%。当前优先评估的 P02/P14/P23 在 pilot 上按三 pair 等权汇总，50% 层为 **{overall_priority.loc[50]:.2f}%**，75% 为 **{overall_priority.loc[75]:.2f}%**，100% 为 **{overall_priority.loc[100]:.2f}%**。100% 是先前固定的统一设置，**并非这三个 pair 的最佳 pilot 层**。full-90 的结论应描述为“固定最终层下的迁移”，不能描述为三个 pair 各自优化过层深后的峰值表现。

![Layer TASR and PSR heatmap](layer_tasr_psr_heatmap.png)

## 关键发现 2：proxy 成功与 target 迁移之间存在明显落差

在 layer 100% pilot 上，三个主 pair 的 PSR 与 TASR 如下；同层 mean CKA 与 target 表示位移一起列出，避免把 proxy 成功等同于迁移成功。

{markdown_table(gap_table, decimals=3)}

P02 与 P23 的 proxy 成功已高，但 target TASR 很低。P14 的 proxy 成功率与 P23 接近，却有更高迁移率。这是当前数据支持的现象；由于 P02 的 proxy 规模和 target family 都不同，不能只用“跨家族”一个因素解释全部差异。

![Pilot geometry versus transfer](pilot_geometry_vs_transfer.png)

## 关键发现 3：CKA 上升不保证最终迁移率上升

跨 {int(layers.tasr_percent.notna().sum())} 个有正条件分母的 pair-layer 聚合点，mean CKA 与条件 TASR 的 Spearman ρ={cka_rho:+.2f}，mean target 表示位移 ΔR 与条件 TASR 的 ρ={shift_rho:+.2f}。这些是描述性关联，层深、pair family 和样本重复都混在一起，不作独立样本显著性结论。具体看 P14：75%→100% 时 CKA 从 0.743 升至 0.900，条件 TASR 从 {pair_depth_rate[('P14', 75)]:.1f}% 变为 {pair_depth_rate[('P14', 100)]:.1f}%；P23 的 CKA 从 0.729 升至 0.834，条件 TASR 从 {pair_depth_rate[('P23', 75)]:.1f}% 变为 {pair_depth_rate[('P23', 100)]:.1f}%。因此 CKA 可反映整体表示对齐，却不能单独预测哪层的攻击梯度最能推动 victim 的目标分类。相应层的 ΔR 也应一起看。

## 关键发现 4：pull+push 的收益不完全均匀

消融使用 P02/P14/P23 的相同 10 个 directions、相同 50 张图/方向。下表 TASR 为百分比。

{markdown_table(ablation_table)}

合并三 pair 且按 proxy 命中数加权时，pull+push 为 {combined_ablation['pull_push']:.2f}%，pull-only 为 {combined_ablation['pull_only']:.2f}%，push-only 为 {push_only_rate}。push-only 的主条件指标不能与另两组数值比较；其 clean-valid 结果仍保留在原始 CSV。各 pair 差异见表，因此方法优势要按 pair 给出，不应写成“每个模型组合都提高”。

![Loss ablation by pair](ablation_by_pair.png)

## 关键发现 5：pilot 的 variance 尚不足以解释方向不对称

在 layer100 的 8 pair × 5 个无序类别对（n=40）中，对 `TASR(A→B)-TASR(B→A)` 与 source−target variance 差做 Spearman，ρ={signed_var_rho:+.2f}；与双向 gap-closure 差为 ρ={signed_gap_rho:+.2f}。这提示类别离散度**单独**不足以解释方向差，而实际攻击造成的 margin gap 变化可能更贴近结果。该 pilot 只有 5 个无序类对/模型 pair，Gemma 两组接近 TASR ceiling，且各点共享类与图片；相关性仍是探索性。full-90 分析将给每个主 pair 增加到 45 个无序类别对，并保留 per-image 数据来检验。

![Pilot asymmetry factors](pilot_asymmetry_factors.png)

各 target encoder 的类别 cosine dispersion 如下。数值是在各自原始表示空间中、对 L2 归一化的 reference embeddings 计算；这里显示结构差异，但不同模型的 PCA/t-SNE 坐标不可直接叠加比较。

![Class variance by target](class_variance_by_target.png)

## 解释边界

这些 pilot 图的 PCA/t-SNE 已有原始输出，但统计结论使用原空间 cosine、CKA/RSA、margin 和 variance；t-SNE 的二维全局距离不作定量证据。layer pilot 的 10 个 directions 承担层选择，不能同时当作独立验证。完整 80 个 heldout directions 将在三 pair full-90 评估结束后列入最终主报告。所有 CSV 与图背后的行级数据保存在本目录和原 `analysis/layers/` 中。
"""
    (DEST / "all_completed_analysis.md").write_text(report, encoding="utf-8")
    with zipfile.ZipFile(DEST.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(DEST.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print(f"wrote complete pilot/ablation synthesis under {DEST}")


if __name__ == "__main__":
    main()
