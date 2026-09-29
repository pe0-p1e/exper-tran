#!/usr/bin/env python3
"""Validate and package the two-cross/one-intra full-90 results and real PNGs."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, get_pair

PAIRS = ("P02", "P14", "P23")
PAIR_NAMES = {
    "P02": "Qwen3.5-4B → Gemma 4 E4B (cross-family)",
    "P14": "Qwen3.5-2B → Qwen3.5-4B (intra-family)",
    "P23": "Qwen3.5-2B → InternVL3.5-4B (cross-family)",
}
ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/a100_final_v5"
EXPERIMENT = Path(__file__).resolve().parents[1]
SHARE = OUTPUT / "shareable/priority3"
SUMMARY_STEM = "single_proxy_full90_priority3"
ANALYSIS = OUTPUT / "analysis" / SUMMARY_STEM


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rate(rows: list[dict[str, str]], hits: str, denominator: str = "images") -> str:
    numerator = sum(int(row[hits]) for row in rows)
    total = sum(int(row[denominator]) for row in rows)
    return f"{100 * numerator / total:.2f}% ({numerator}/{total})" if total else "— (0/0)"


def conditional_rate(rows: list[dict[str, str]]) -> str:
    return rate(rows, "eligible_tasr_hits", "eligible_tasr_denominator")


def percent_or_na(value: str) -> str:
    return f"{100 * float(value):.1f}%" if value else "—"


def add_file(archive: zipfile.ZipFile, source: Path, destination: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.write(source, destination)


def main() -> None:
    SHARE.mkdir(parents=True, exist_ok=True)
    from analyze_all_completed import main as refresh_completed_analysis
    from export_all_measurements import main as refresh_all_measurements
    from analyze_during_evaluation import main as refresh_paired_analysis

    refresh_completed_analysis()
    refresh_all_measurements()
    refresh_paired_analysis()
    refresh_all_measurements()
    summaries = OUTPUT / "summaries"
    full = read_csv(summaries / f"{SUMMARY_STEM}.csv")
    pilot = read_csv(summaries / f"{SUMMARY_STEM}_pilot.csv")
    heldout = read_csv(summaries / f"{SUMMARY_STEM}_heldout80.csv")
    expected = {(pair, f"T{index:02d}") for pair in PAIRS for index in range(1, 91)}
    if len(full) != 270 or {(row["pair_id"], row["transition_id"]) for row in full} != expected:
        raise RuntimeError("Priority full-90 summary must contain exactly 270 distinct cells")
    if len(pilot) != 30 or len(heldout) != 240:
        raise RuntimeError("Pilot/heldout split must contain 30/240 cells")
    if any(int(row["images"]) != 50 for row in full):
        raise RuntimeError("Every main cell must have 50 clean-valid images")
    if {(row["pair_id"], row["transition_id"]) for row in pilot + heldout} != expected:
        raise RuntimeError("Pilot/heldout rows do not cover the full main set")

    geometry = {}
    figure_dir = SHARE / "figures"
    figure_dir.mkdir(exist_ok=True)
    for pair in PAIRS:
        directory = ANALYSIS / pair
        shift = read_csv(directory / "representation_shift.csv")
        variance = read_csv(directory / "class_variance.csv")
        asymmetry = read_csv(directory / "asymmetry.csv")
        correlations = read_csv(directory / "correlation_matrix.csv")
        if len(shift) != 90 or len(variance) != 10 or len(asymmetry) != 90:
            raise RuntimeError(f"Incomplete full-90 geometry for {pair}")
        geometry[pair] = (shift, variance, asymmetry, correlations)
        unordered_ids = {
            row["transition_id"]
            for row in full
            if row["pair_id"] == pair and int(row["source"]) < int(row["target"])
        }
        if len(unordered_ids) != 45 or any(
            not (directory / kind / f"{pair}_{transition_id}.png").is_file()
            for transition_id in unordered_ids
            for kind in ("pca", "tsne")
        ):
            raise RuntimeError(f"Expected joint PCA/t-SNE for all 45 unordered class pairs: {pair}")
        for kind in ("pca", "tsne"):
            source = directory / kind / f"{pair}_T01.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, figure_dir / f"{pair}_{kind}.png")
        shutil.copy2(directory / "correlation_matrix.png", figure_dir / f"{pair}_correlation.png")

    examples_dir = SHARE / "examples"
    examples = read_csv(examples_dir / "examples.csv")
    if len(examples) != 6 or {row["pair_id"] for row in examples} != set(PAIRS):
        raise RuntimeError("Expected one real target hit and miss for every priority pair")
    for row in examples:
        for column in ("clean_png", "adversarial_png"):
            if not (examples_dir / row[column]).is_file():
                raise FileNotFoundError(examples_dir / row[column])

    layer_lines = []
    priority_layer_rates = {}
    for depth in ("001", "025", "050", "075", "100"):
        rows = read_csv(summaries / f"layer_{depth}_results.csv")
        if len(rows) != 80:
            raise RuntimeError(f"Incomplete layer pilot: {depth}")
        layer_lines.append(
            f"| {int(depth)}% | {conditional_rate(rows)} | {rate(rows, 'proxy_hits')} | "
            f"{rate(rows, 'asr_hits')} |"
        )
        priority_layer_rates[int(depth)] = conditional_rate(
            [row for row in rows if row["pair_id"] in PAIRS]
        )
    ablation_lines = []
    for arm, name in (("pull_only", "Pull only"), ("push_only", "Push only"), ("pull_push", "Pull + push")):
        rows = read_csv(summaries / f"ablation_{arm}.csv")
        if len(rows) != 30 or {row["pair_id"] for row in rows} != set(PAIRS):
            raise RuntimeError(f"Ablation arm is incomplete: {arm}")
        ablation_lines.append(f"| {name} | {conditional_rate(rows)} | {rate(rows, 'asr_hits')} |")

    def subset(rows: list[dict[str, str]], pair: str) -> list[dict[str, str]]:
        return [row for row in rows if row["pair_id"] == pair]

    main_lines = [
        f"| {pair} | {PAIR_NAMES[pair]} | {conditional_rate(subset(full, pair))} | "
        f"{conditional_rate(subset(heldout, pair))} | "
        f"{rate(subset(full, pair), 'proxy_hits')} | "
        f"{rate(subset(full, pair), 'asr_hits')} |"
        for pair in PAIRS
    ]
    combined_line = (
        f"| Three-pair micro total | — | {conditional_rate(full)} | "
        f"{conditional_rate(heldout)} | {rate(full, 'proxy_hits')} | "
        f"{rate(full, 'asr_hits')} |"
    )

    direction_lines = []
    asymmetry_lines = []
    correlation_lines = []
    for pair in PAIRS:
        shift, _variance, asymmetry, correlations = geometry[pair]
        for direction in ("T01", "T02"):
            row = next(row for row in shift if row["transition_id"] == direction)
            direction_lines.append(
                f"| {pair} | {direction} ({row['source_label']}→{row['target_label']}) | "
                f"{percent_or_na(row['TASR'])} | {float(row['mean_delta_R']):+.4f} | "
                f"{float(row['clean_target_margin']):+.4f} | "
                f"{float(row['source_variance']):.4f} | "
                f"{float(row['target_variance']):.4f} | "
                f"{float(row['prototype_distance']):.4f} |"
            )
        unique = [
            row for row in asymmetry
            if int(row["source_label"]) < int(row["target_label"]) and row["delta_TASR"]
        ]
        unique.sort(key=lambda row: abs(float(row["delta_TASR"])), reverse=True)
        for row in unique[:3]:
            asymmetry_lines.append(
                f"| {pair} | {row['source_label']}→{row['target_label']} | "
                f"{100 * float(row['TASR_forward']):.1f}% | "
                f"{100 * float(row['TASR_reverse']):.1f}% | "
                f"{100 * float(row['delta_TASR']):+.1f} pp | "
                f"{float(row['variance_difference']):+.4f} | "
                f"{float(row['clean_margin_difference']):+.4f} |"
            )
        wanted = ("CKA", "RSA", "source_variance", "target_variance", "mean_delta_R", "clean_target_margin", "gap_closure")
        values = {
            row["metric_y"]: row
            for row in correlations
            if row["metric_x"] == "TASR" and row["metric_y"] in wanted
        }
        correlation_lines.append(
            f"| {pair} | " + " | ".join(
                f"{float(values[key]['spearman_rho']):+.2f}" if key in values else "n/a"
                for key in wanted
            ) + " |"
        )

    figure_lines = [
        f"| {pair} | ![PCA]({{figure}}) | ![t-SNE]({{tsne}}) |".format(
            figure=f"figures/{pair}_pca.png", tsne=f"figures/{pair}_tsne.png"
        )
        for pair in PAIRS
    ]
    example_lines = [
        f"| {row['pair_id']} | {'hit' if row['target_hit'] == 'True' else 'miss'} | "
        f"![clean](examples/{row['clean_png']}) | "
        f"![adversarial](examples/{row['adversarial_png']}) |"
        for row in examples
    ]

    report = f"""# A100 v5：两组跨家族、一组同家族的定向迁移实验报告

**数据状态：完整。** 本轮主评估覆盖 3 个 proxy→target 组合、90 个有向类别迁移/组合、50 张共同 clean-valid 图片/迁移，共 13,500 个 target 评估样本。P16/P19/P20/P21/P22 的已有数据保留，但不混入此主结果。

## 研究问题与设计

检验基于 proxy 视觉表示的语义 pull/push 攻击能否迁移到 target，并用两个跨家族 pair 与一个同家族 pair 比较。P14 和 P23 共享 **Qwen3.5-2B proxy**，分别指向 Qwen3.5-4B 和 InternVL3.5-4B；这提供了在同一个攻击源下比较同家族与跨家族 target 的机会。P02 提供第二个跨家族模型组合。

10 个 ImageNet diverse 类为 goldfish、monarch butterfly、pineapple、acoustic guitar、laptop、espresso、volcano、rocking chair、soccer ball、school bus。每类 48 张 reference images，另从固定的 64 张 train candidates 里筛出 50 张共同 clean-valid 攻击图；全部 10×9=90 个有向非 self-transition 使用同一 canonical manifest。攻击从预先冻结的 clean 图片生成 PNG，target 仅对已写好的 PNG 评估，没有 target 反馈进入梯度或参数选择。

攻击配置：BF16 proxy、vision encoder mean pooling、以 10-direction pilot 选出的最终视觉层、seed=42、随机起点、epsilon=16/255、50 steps、step size=1/255、momentum=1。分类 loss 权重 `lambda_cls=0`，优化循环跳过 language classification forward。语义 loss 在所有主 cell 固定为

\\[L=1.5(1-\\cos(z_{{adv}},\\mu_t))+0.5(1+\\cos(z_{{adv}},\\mu_s)).\\]

主 TASR 以 proxy 已命中目标类的图像为分母，分子是其中 target 也命中的图像；proxy 未命中者不计主分母。PSR 仍以全部 clean-valid 图像为分母，N=50/cell。原 clean-valid 分母的 `unconditional_tasr_percent` 同时保存在 CSV 供复核。条件分母随 cell 变化，聚合率按命中数与分母求和，不能平均各 cell 百分比。

## 主实验：full-90 评估

| Pair | Proxy → target | TASR 全90 | TASR heldout80 | PSR 全90 | ASR 全90 |
|---|---|---:|---:|---:|---:|
{chr(10).join(main_lines)}
{combined_line}

10 个 pilot directions 用于选择层；其余 80 个 directions 没有参与层选择，因此 heldout80 是更合适的泛化检查。三 pair pilot micro TASR = **{conditional_rate(pilot)}**；heldout80 micro TASR = **{conditional_rate(heldout)}**。各 cell 的 source、target、命中次数、运行设置和双口径指标见随包 CSV。

## 层深实验与方法消融

层深 pilot 覆盖原始 8 个 pair × 10 directions × 50 张图/层（每层 N=4,000）。每个层百分比通过实际 vision block 数映射，具体 block ID 见审计/配置表。

| 视觉层深度 | TASR | PSR | ASR |
|---:|---:|---:|---:|
{chr(10).join(layer_lines)}

先前按 clean-valid 分母比较原 8-pair pilot，100% 层为 43.625%，75% 为 42.125%，因此固定了 100% 层。按当前 proxy 条件分母重新报告时，当前三 pair 的 50% 层为 {priority_layer_rates[50]}，75% 层为 {priority_layer_rates[75]}，100% 层为 {priority_layer_rates[100]}；**最终层并非这三 pair 的 pilot 最优层**。本轮 full-90 是统一固定最终层的公平比较，不能写成三 pair 各自最优层的结果。该 pilot 本身承担层选择，也不应用作独立验证。

loss 消融使用本轮相同的 P02/P14/P23、10 个 pilot directions、每 arm 1,500 张，攻击预算相同：

| Loss | TASR | ASR |
|---|---:|---:|
{chr(10).join(ablation_lines)}

这些是样本加权的描述性比例。50 张图共享同一类别迁移和模型，不能把它们当成彼此独立的 50 次实验；后续显著性评估应按 transition 或类别聚类。

## 表示空间距离、降维与方向不对称

每个 target encoder 对 10 类 references、clean 和 adversarial PNG 抽取同一空间的表示，L2 归一化后计算目标/源 prototype cosine、类别离散度、协方差 trace、effective rank、CKA、RSA、clean margin、margin change 和 gap closure。`mean_delta_R` 是 target prototype pull 与 source prototype push 的和。下面以 goldfish ↔ monarch butterfly 为例，展示两个方向的原空间指标；同一 pair 内 prototype distance 对调 source/target 后不变。

| Pair | Direction | TASR | ΔR | Clean target margin | Source variance | Target variance | Prototype distance |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(direction_lines)}

Joint PCA/t-SNE 对每个 target 模型的 **45 个无序类别对**分别制作一套图：每套图把这两个类的 references 与**两个方向**的 clean/adv 图片拼接后只 fit 一次。PCA 上标出 clean→adv 位移；t-SNE 仅作定性局部结构图。不同模型或不同类别对的图各有自己的降维坐标系，坐标距离不可跨图直接比较；下表的定量距离、方差、margin 都来自原始 embedding 空间。此处展示 goldfish ↔ monarch butterfly 的代表图，全部 270 张 PCA/t-SNE 图在归档 `results/analysis/` 中。

| Pair | Joint PCA | Joint t-SNE |
|---|---|---|
{chr(10).join(figure_lines)}

每个 pair 的无序类别对里，TASR 双向差最大的三个如下。Δvariance 是 source 类减 target 类参考云的 cosine dispersion，Δclean margin 是两个方向的初始 target margin 之差。Prototype distance 对两个方向完全相同，因此单靠它无法解释 TASR 方向差；方差、初始 margin、实际表示位移与决策边界共同值得检查。这里的关联不构成因果证明。

| Pair | A→B | TASR A→B | TASR B→A | Δ TASR | Δ variance | Δ clean margin |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(asymmetry_lines)}

以下 Spearman ρ 在每个 pair 的 90 个 directed transitions 上计算，衡量 TASR 与原空间几何统计的单调关联。类别方向存在配对与共享图像，因此这些 ρ 是探索性结果。

| Pair | CKA | RSA | Source var | Target var | ΔR | Clean margin | Gap closure |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(correlation_lines)}

完整 `correlation_matrix.csv`/热图、class variance、per-image representation shift 与 asymmetry CSV 均在归档内。t-SNE 二维欧氏距离没有用于相关性或机制结论。

## 真实攻击图片示例

下列 T01（goldfish→monarch butterfly）图片全部取自实验保存的 PNG。每个 pair 展示一组目标命中和一组未命中；`examples.csv` 保存目标模型的 clean/adv 解析标签。所有展示样例的 clean→adv 最大像素变化为 16/255。

| Pair | Target outcome | Clean PNG | Adversarial PNG |
|---|---|---|---|
{chr(10).join(example_lines)}

## 可复现文件与限制

结果包包含 main full-90、pilot10、heldout80 三张主表、5 层 pilot、3 组 ablation、每个 pair 的 full-90 geometry 和相关矩阵、canonical manifests、clean screen、模型/层审计、YAML 配置、运行和分析脚本、环境/model revision 元数据，以及真实 PNG 样例。模型权重和 ImageNet 原始图像不在压缩包内；manifest 记录数据选择，PNG 示例可直接查看。

本轮选择 P14/P23 共享 proxy，有助于比较 target family；P02 的 proxy 规模不同，因此跨两个跨家族 pair 直接归因于 family 时要考虑 proxy 差异。主 TASR 使用 proxy-success 条件分母，可能因为 proxy 失败样本被排除而高于全 clean-valid 率；两种口径均在 CSV 中。若之后补做 P16/P19/P20/P21/P22，应沿用当前 manifest、攻击预算和指标定义。

已完成的全部 8-pair × 5-layer pilot 与三组消融另有一份综合解释：`completed_data_analysis/all_completed_analysis.md`。它包含 pair 级图表、CKA/表示位移与迁移率的关系，以及 pilot 方差对不对称性解释的检验；其中未完成的 full-90 cell 明确排除在结论之外。
"""
    report_path = SHARE / "experiment_report.md"
    report_path.write_text(report, encoding="utf-8")

    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": list(PAIRS),
        "scope": "two cross-family, one intra-family",
        "transitions_per_pair": 90,
        "clean_valid_images_per_cell": 50,
        "total_main_images": 13500,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_revisions": {
            model: MODEL_REVISIONS[model]
            for pair in PAIRS
            for model in (get_pair(pair).proxy_model, get_pair(pair).target_model)
        },
    }
    metadata_path = SHARE / "environment_and_scope.json"
    metadata_path.write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n")

    archive_path = SHARE / "a100_final_v5_priority3_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        add_file(archive, report_path, "experiment_report.md")
        add_file(archive, metadata_path, "environment_and_scope.json")
        measurement_archive = SHARE / "all_available_measurements.zip"
        if measurement_archive.is_file():
            add_file(archive, measurement_archive, "all_available_measurements.zip")
        for path in sorted(examples_dir.iterdir()):
            add_file(archive, path, f"examples/{path.name}")
        for path in sorted(figure_dir.glob("*.png")):
            add_file(archive, path, f"figures/{path.name}")
        completed_analysis = SHARE / "completed_data_analysis"
        for path in sorted(completed_analysis.rglob("*")):
            if path.is_file():
                add_file(
                    archive, path,
                    f"completed_data_analysis/{path.relative_to(completed_analysis)}",
                )
        concurrent_analysis = SHARE / "concurrent_analysis"
        for path in sorted(concurrent_analysis.rglob("*")):
            if path.is_file():
                add_file(archive, path, f"concurrent_analysis/{path.relative_to(concurrent_analysis)}")
        for path in sorted(summaries.glob("layer_*_results.csv")):
            add_file(archive, path, f"results/summaries/{path.name}")
        for path in sorted(summaries.glob("ablation_*.csv")):
            add_file(archive, path, f"results/summaries/{path.name}")
        for name in (f"{SUMMARY_STEM}.csv", f"{SUMMARY_STEM}_pilot.csv", f"{SUMMARY_STEM}_heldout80.csv"):
            add_file(archive, summaries / name, f"results/summaries/{name}")
        for pair in PAIRS:
            for path in sorted((ANALYSIS / pair).rglob("*")):
                if path.is_file():
                    add_file(archive, path, f"results/analysis/{pair}/{path.relative_to(ANALYSIS / pair)}")
        for path in sorted((OUTPUT / "calibration").glob("*.csv")):
            add_file(archive, path, f"results/calibration/{path.name}")
        for name in ("analysis/model_layer_audit.csv", "diagnostics/clean_screen_summary.csv"):
            path = OUTPUT / name
            if path.is_file():
                add_file(archive, path, f"results/{path.name}")
        for path in sorted((OUTPUT / "evaluation/manifests").rglob("*.jsonl")):
            add_file(archive, path, f"manifests/{path.relative_to(OUTPUT / 'evaluation/manifests')}")
        for path in (
            EXPERIMENT / "config/full90_selected_layer_priority3.yaml",
            EXPERIMENT / "config/full90_selected_layer_main5.yaml",
            EXPERIMENT / "run_main5_finish.sh",
            EXPERIMENT / "run_priority3_finish.sh",
            EXPERIMENT / "src/run_a100_final.py",
            EXPERIMENT / "src/analyze_joint_geometry.py",
            EXPERIMENT / "src/split_full90_summary.py",
            EXPERIMENT / "src/make_priority3_examples.py",
            EXPERIMENT / "src/queue_priority3_precompute.py",
            EXPERIMENT / "src/analyze_all_completed.py",
            EXPERIMENT / "src/export_all_measurements.py",
            EXPERIMENT / "src/analyze_during_evaluation.py",
            Path(__file__).resolve(),
        ):
            add_file(archive, path, f"settings_and_code/{path.relative_to(EXPERIMENT)}")
    print(f"wrote archive={archive_path} bytes={archive_path.stat().st_size}")


if __name__ == "__main__":
    main()
