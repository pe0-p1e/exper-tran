#!/usr/bin/env python3
"""Report coverage and descriptive geometry for deferred full-90 pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/a100_final_v5"
GEOMETRY = BASE / "analysis/single_proxy_full90_partial"
DEST = BASE / "shareable/priority3/all_available_measurements"
PAIRS = ("P16", "P19", "P20", "P21", "P22")


def corr(frame: pd.DataFrame, left: str, right: str) -> tuple[float, int]:
    observed = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(observed) < 3 or observed[left].nunique() < 2 or observed[right].nunique() < 2:
        return np.nan, len(observed)
    return float(spearmanr(observed[left], observed[right]).statistic), len(observed)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in PAIRS:
        path = GEOMETRY / pair / "representation_shift.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for metric in ("CKA", "RSA", "mean_delta_R", "source_variance", "target_variance", "clean_target_margin", "gap_closure"):
            rho, n = corr(frame, "TASR", metric)
            rows.append({
                "pair_id": pair,
                "metric": metric,
                "spearman_rho_with_conditional_TASR": rho,
                "n_evaluated_cells_with_defined_TASR": n,
                "n_attack_cells_with_geometry": len(frame),
                "n_target_evaluated_cells": int((frame.N_target_evaluated > 0).sum()),
                "n_proxy_success_images": int(frame.N_proxy_success.sum()),
                "mean_metric_all_attack_cells": frame[metric].mean(),
            })
    columns = [
        "pair_id", "metric", "spearman_rho_with_conditional_TASR",
        "n_evaluated_cells_with_defined_TASR", "n_attack_cells_with_geometry",
        "n_target_evaluated_cells", "n_proxy_success_images", "mean_metric_all_attack_cells",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(DEST / "partial_geometry_correlations.csv", index=False)
    lines = [
        "# 暂缓组合的部分 full-90 表示空间测量", "",
        "这些结果只用已冻结的攻击 PNG，未完成 target 分类的 cell 仍可测 CKA、RSA、class variance、prototype distance、Δpull、Δpush、ΔR、margin、gap closure。其 TASR 留空。条件 TASR 的分母只含 proxy 命中图像；分母为 0 时未定义。", "",
        "未完成 pair 的 target 评估是按运行队列顺序产生的部分样本，相关系数只作探索性描述，不能推断完整 90 directions。定量距离来自原始 embedding，joint t-SNE 只作可视化。", "",
        "| Pair | 几何 cell | 已评估 cell | proxy 命中图像 | CKA–TASR ρ (n) | ΔR–TASR ρ (n) |", "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in PAIRS:
        subset = [r for r in rows if r["pair_id"] == pair]
        if not subset:
            lines.append(f"| {pair} | 待测 | — | — | — | — |")
            continue
        by_metric = {r["metric"]: r for r in subset}
        def cell(metric):
            item = by_metric[metric]
            rho = item["spearman_rho_with_conditional_TASR"]
            return f"{rho:+.2f} ({item['n_evaluated_cells_with_defined_TASR']})" if np.isfinite(rho) else f"— ({item['n_evaluated_cells_with_defined_TASR']})"
        first = subset[0]
        lines.append(f"| {pair} | {first['n_attack_cells_with_geometry']} | {first['n_target_evaluated_cells']} | {first['n_proxy_success_images']} | {cell('CKA')} | {cell('mean_delta_R')} |")
    lines.extend(["", "逐 cell 指标在 `partial_full90_geometry/Pxx/representation_shift.csv`，逐图指标在 `per_image_representation_shift.csv`；完整的 Spearman 行表在 `partial_geometry_correlations.csv`。", ""])
    (DEST / "partial_geometry_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"partial geometry pairs measured={len({row['pair_id'] for row in rows})}")


if __name__ == "__main__":
    main()
