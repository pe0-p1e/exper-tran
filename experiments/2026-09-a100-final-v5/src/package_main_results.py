#!/usr/bin/env python3
"""Build a compact shareable archive of final results, settings, manifests and examples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import torch
import yaml

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, get_pair


PAIR_IDS = ("P02", "P14", "P16", "P19", "P23")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add_file(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.is_file():
        archive.write(path, arcname)


def fmt_rate(hits: int, total: int) -> str:
    return f"{100 * hits / total:.2f}% ({hits}/{total})" if total else "n/a"


def finish_report(root: Path, output: Path, pair_ids: tuple[str, ...]) -> None:
    """Replace the pending report section with numbers and figures from completed outputs."""
    report_path = output / "experiment_report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## 8. 最终结果补充区"
    completed_marker = "## 8. 最终 main-5 结果"
    if marker not in report and completed_marker in report:
        marker = completed_marker
    if marker not in report:
        raise RuntimeError(f"Report section marker missing: {report_path}")
    all_rows = csv_rows(root / "summaries/single_proxy_full90_main5.csv")
    pilot_rows = csv_rows(root / "summaries/single_proxy_full90_main5_pilot.csv")
    heldout_rows = csv_rows(root / "summaries/single_proxy_full90_main5_heldout80.csv")
    if (len(all_rows), len(pilot_rows), len(heldout_rows)) != (450, 50, 400):
        raise RuntimeError("Main/pilot/heldout summary row counts must be 450/50/400")

    def aggregate(rows: list[dict[str, str]]) -> tuple[int, int, int, int, int]:
        return (
            sum(int(row["tasr_hits"]) for row in rows),
            sum(int(row["images"]) for row in rows),
            sum(int(row["proxy_hits"]) for row in rows),
            sum(int(row["asr_hits"]) for row in rows),
            len(rows),
        )

    def scope_line(name: str, rows: list[dict[str, str]]) -> str:
        hits, total, proxy_hits, asr_hits, cells = aggregate(rows)
        return (
            f"| {name} | {cells} | {total} | {fmt_rate(hits, total)} | "
            f"{fmt_rate(proxy_hits, total)} | {fmt_rate(asr_hits, total)} |"
        )

    pair_lines = []
    for pair_id in pair_ids:
        subset = [row for row in all_rows if row["pair_id"] == pair_id]
        pilot_subset = [row for row in pilot_rows if row["pair_id"] == pair_id]
        heldout_subset = [row for row in heldout_rows if row["pair_id"] == pair_id]
        pair_lines.append(
            f"| {pair_id} | {scope_line('all', subset).split('|')[4].strip()} | "
            f"{scope_line('pilot', pilot_subset).split('|')[4].strip()} | "
            f"{scope_line('heldout', heldout_subset).split('|')[4].strip()} | "
            f"{scope_line('all', subset).split('|')[5].strip()} | "
            f"{scope_line('all', subset).split('|')[6].strip()} |"
        )

    def metric_correlations(pair_id: str) -> list[str]:
        path = root / "analysis/single_proxy_full90_main5" / pair_id / "correlation_matrix.csv"
        if not path.is_file():
            return []
        rows = csv_rows(path)
        wanted = {"CKA", "RSA", "source_variance", "target_variance", "mean_delta_R", "clean_target_margin", "gap_closure"}
        return [
            f"{row['metric_y']} ρ={float(row['spearman_rho']):.3f} (n={row['n']})"
            for row in rows
            if row.get("metric_x") == "TASR" and row.get("metric_y") in wanted
        ]

    asymmetry_lines = []
    correlation_lines = []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for pair_id in pair_ids:
        analysis_dir = root / "analysis/single_proxy_full90_main5" / pair_id
        shift_path = analysis_dir / "representation_shift.csv"
        if len(csv_rows(shift_path)) != 90:
            raise RuntimeError(f"Full-90 representation analysis is incomplete: {shift_path}")
        asym_path = analysis_dir / "asymmetry.csv"
        asym_rows = csv_rows(asym_path) if asym_path.is_file() else []
        comparable = [row for row in asym_rows if row.get("TASR_forward") not in (None, "")]
        comparable.sort(key=lambda row: abs(float(row["delta_TASR"])), reverse=True)
        extremes = comparable[:2]
        if extremes:
            for row in extremes:
                asymmetry_lines.append(
                    f"| {pair_id} | {row['source_label']}→{row['target_label']} | "
                    f"{float(row['TASR_forward']):.2f} | {float(row['TASR_reverse']):.2f} | "
                    f"{float(row['delta_TASR']):+.2f} | {float(row['variance_difference']):+.4f} | "
                    f"{float(row['clean_margin_difference']):+.4f} |"
                )
        correlations = metric_correlations(pair_id)
        if correlations:
            correlation_lines.append(f"- **{pair_id}:** " + "; ".join(correlations))
        for kind in ("pca", "tsne"):
            source = analysis_dir / kind / f"{pair_id}_T01.png"
            if source.is_file():
                shutil.copy2(source, figure_dir / f"{pair_id}_{kind}.png")

    pilot_total = aggregate(pilot_rows)
    heldout_total = aggregate(heldout_rows)
    all_total = aggregate(all_rows)
    metric_text = "\n".join(correlation_lines) if correlation_lines else "- 尚未发现可读取的 full-90 correlation matrix。"
    asym_text = "\n".join(asymmetry_lines) if asymmetry_lines else "| — | — | — | — | — | — | — |"
    new_section = f'''## 8. 最终 main-5 结果

**评估已完成：** {all_total[4]} 个 pair-transition cell，{all_total[1]} 张 clean-valid 图。攻击生成先于 target 评估；P20/P21/P22 未混入本轮 main-5。主 TASR 按 clean-valid 图计算。

| 范围 | Cells | N | TASR | PSR | ASR |
|---|---:|---:|---:|---:|---:|
{scope_line("Pilot 10 directions", pilot_rows)}
{scope_line("Held-out 80 directions", heldout_rows)}
{scope_line("All 90 directions", all_rows)}

| Pair | TASR 全90 | TASR pilot-10 | TASR heldout-80 | PSR 全90 | ASR 全90 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pair_lines)}

### 方向不对称性

每个 pair 列出 TASR 绝对差最大的两个无序类别对。这里的差值定义为表中顺序 `source→target` 的 TASR 减反向 TASR；variance difference 为 source 类参考云离散度减 target 类离散度。它们是待解释的关联特征，并不证明方差导致不对称。

| Pair | Direction | TASR forward | TASR reverse | Δ TASR | Δ variance | Δ clean margin |
|---|---|---:|---:|---:|---:|---:|
{asym_text}

### TASR 与几何指标的 Spearman 相关

相关矩阵按 pair 的 90 个 directed transitions 计算；只报告分析 CSV 中可用指标。相关性跨方向汇总，不能作因果解释，且不同 transition 不是完全独立样本。

{metric_text}

### Joint PCA 与 t-SNE 图

每张图在各自 target encoder 的同一表示空间内联合拟合 references 与两个方向的 clean/adversarial 数据。不同模型图的坐标轴不可互相比较。t-SNE 只作定性展示。

| Pair | Joint PCA | Joint t-SNE |
|---|---|---|
{chr(10).join(f"| {p} | ![PCA]({(Path('figures') / f'{p}_pca.png').as_posix()}) | ![t-SNE]({(Path('figures') / f'{p}_tsne.png').as_posix()}) |" for p in pair_ids)}

主表和完整 per-cell 数据见随包 CSV；PNG 成败样例见 `examples/`，其 `examples.csv` 给出真实类别方向和判定结果。Pilot-layer 与 ablation 仍按前文给出的汇总数值报告，完整行级数据已随包。
'''
    report_path.write_text(report.split(marker)[0] + new_section, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/a100_final_v5/shareable"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = repo / "outputs/a100_final_v5"
    exp = repo / "experiments/2026-09-a100-final-v5"
    main_summary = root / "summaries/single_proxy_full90_main5.csv"
    rows = csv_rows(main_summary)
    expected = {(pair, f"T{index:02d}" if index <= 10 else f"T{index:02d}") for pair in PAIR_IDS for index in range(1, 91)}
    observed = {(row["pair_id"], row["transition_id"]) for row in rows}
    if len(rows) != 450 or observed != expected:
        raise RuntimeError(f"Final main summary is incomplete: {len(rows)}/450 rows")
    for name in ("single_proxy_full90_main5_pilot.csv", "single_proxy_full90_main5_heldout80.csv"):
        if not (root / "summaries" / name).is_file():
            raise FileNotFoundError(root / "summaries" / name)

    main_config_path = exp / "config/full90_selected_layer_main5.yaml"
    main_config = yaml.safe_load(main_config_path.read_text(encoding="utf-8"))
    classes = {int(item["label"]): str(item["name"]) for item in main_config["classes"]}
    examples_dir = output / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_rows = []

    # Pick one successful and one failed targeted example per evaluated model pair.
    for pair_id in PAIR_IDS:
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        selected = []
        for outcome, ordered_rows, desired in (
            ("hit", sorted(pair_rows, key=lambda row: -int(row["tasr_hits"])), True),
            ("miss", sorted(pair_rows, key=lambda row: int(row["tasr_hits"])), False),
        ):
            for row in ordered_rows:
                transition_id = row["transition_id"]
                state_path = root / "states_layer_100" / pair_id / transition_id / "batch_00.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                mask = state["target"]["target_hit_mask"]
                indices = [index for index, hit in enumerate(mask) if bool(hit) is desired]
                if indices:
                    selected.append((outcome, row, state, indices[0]))
                    break
            if not selected and outcome == "hit":
                print(f"no target-hit example available for {pair_id}", file=sys.stderr)
        for outcome, row, state, image_index in selected:
            transition_id = row["transition_id"]
            image_dir = (
                root
                / "attacks"
                / pair_id
                / f"a100_v5_{transition_id}"
                / "batch_00"
                / str(state["objective_tag"])
                / f"lambda_{float(state.get('lambda_cka', 1.0)):g}"
            )
            base = f"{pair_id}_{transition_id}_class{row['source']}_to_{row['target']}_{outcome}"
            clean_src = image_dir / f"{image_index:02d}_clean.png"
            adv_src = image_dir / f"{image_index:02d}_adv.png"
            if not clean_src.is_file() or not adv_src.is_file():
                raise FileNotFoundError(f"Missing PNG example for {pair_id}/{transition_id}")
            clean_name, adv_name = f"{base}_clean.png", f"{base}_adv.png"
            shutil.copy2(clean_src, examples_dir / clean_name)
            shutil.copy2(adv_src, examples_dir / adv_name)
            adv_output = state["target"]["adversarial_outputs"][image_index]
            example_rows.append(
                {
                    "pair_id": pair_id,
                    "proxy_model": get_pair(pair_id).proxy_model,
                    "target_model": get_pair(pair_id).target_model,
                    "transition_id": transition_id,
                    "source_label": row["source"],
                    "source_class": classes[int(row["source"])],
                    "target_label": row["target"],
                    "target_class": classes[int(row["target"])],
                    "example_outcome": outcome,
                    "adversarial_parsed_label": adv_output.get("parsed_label"),
                    "clean_png": clean_name,
                    "adversarial_png": adv_name,
                }
            )
    examples_csv = examples_dir / "examples.csv"
    with examples_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(example_rows[0]))
        writer.writeheader()
        writer.writerows(example_rows)

    metadata = {
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "main_evaluated_pairs": list(PAIR_IDS),
        "deferred_reverse_scale_pairs": ["P20", "P21", "P22"],
        "selected_layer_percent": int(main_config["pilot_selection_layer_percent"]),
        "transitions": 90,
        "attack_images_per_cell": int(main_config["attack_count"]),
        "batches_per_cell": int(main_config["attack_count"] / main_config["batch_size"]),
        "model_revisions": {model: MODEL_REVISIONS[model] for pair_id in PAIR_IDS for model in (get_pair(pair_id).proxy_model, get_pair(pair_id).target_model)},
        "excluded_pairs_note": "P20/P21 attacks were generated and retained; P22 was stopped at 58/90. All three are excluded from this round's main target evaluation and can be evaluated later.",
    }
    metadata_path = output / "environment_and_scope.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    readme = output / "README.md"
    readme.write_text(
        "# A100 final experiment results\n\n"
        "This bundle contains the completed layer pilot, controlled loss ablation, "
        "the selected 100% vision-layer full-90 evaluation for five model pairs, "
        "representative clean/adversarial PNG pairs, manifests, diagnostics, and run settings.\n\n"
        "## Main protocol\n\n"
        "- 10 ImageNet classes; all 90 directed non-self source→target transitions.\n"
        "- 50 common clean-valid images per transition; 50 attack steps; epsilon 16/255; "
        "step size 1/255; momentum 1; random start; BF16 proxy.\n"
        "- Linear pull-push loss: target weight 1.5, source weight 0.5; lambda_cls=0.\n"
        "- Layer selected by micro TASR over the 10-transition pilot; selected layer is 100%.\n"
        "- P20/P21/P22 reverse-scale pairs are deferred from the main evaluation. P20/P21 "
        "attack outputs are retained; P22 stopped at 58/90 transitions.\n"
        "- `single_proxy_full90_main5_pilot.csv` contains the 10 layer-selection transitions; "
        "`single_proxy_full90_main5_heldout80.csv` contains the remaining 80 transitions.\n\n"
        "## Metric\n\n"
        "TASR is targeted success divided by all clean-valid images (N=50 per cell). "
        "PSR is proxy targeted success over the same denominator.\n\n"
        "See `configs/full90_selected_layer_main5.yaml` and `results/` for the exact settings "
        "and outputs. The `examples/` folder pairs each clean PNG with its adversarial PNG; "
        "`examples.csv` records model pair, class direction, and target hit/miss.\n",
        encoding="utf-8",
    )

    finish_report(root, output, PAIR_IDS)

    archive_path = output / "a100_final_v5_main_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        add_file(archive, readme, "README.md")
        add_file(archive, output / "experiment_report.md", "experiment_report.md")
        add_file(archive, metadata_path, "environment_and_scope.json")
        for path in [
            exp / "config/a100_final.yaml",
            main_config_path,
            *sorted((exp / "config/layers").glob("layer_*.yaml")),
            *sorted(exp.glob("config/ablation_*.yaml")),
            exp / "run_8h_experiment.sh",
            exp / "run_main5_finish.sh",
            exp / "src/select_pilot_layer.py",
            exp / "src/split_full90_summary.py",
        ]:
            add_file(archive, path, f"configs_and_runner/{path.relative_to(exp)}")
        for path in sorted((root / "summaries").glob("layer_*_results.csv")):
            add_file(archive, path, f"results/summaries/{path.name}")
        for path in sorted((root / "summaries").glob("ablation_*.csv")):
            add_file(archive, path, f"results/summaries/{path.name}")
        for name in (
            "single_proxy_full90_main5.csv",
            "single_proxy_full90_main5_pilot.csv",
            "single_proxy_full90_main5_heldout80.csv",
        ):
            add_file(archive, root / "summaries" / name, f"results/summaries/{name}")
        for path in sorted((root / "analysis").rglob("*")):
            if path.is_file() and "embeddings" not in path.parts and "smoke" not in path.parts:
                add_file(archive, path, f"results/analysis/{path.relative_to(root / 'analysis')}")
        for path in sorted((root / "calibration").glob("*.csv")):
            add_file(archive, path, f"results/calibration/{path.name}")
        add_file(archive, root / "analysis/model_layer_audit.csv", "results/model_layer_audit.csv")
        add_file(archive, root / "diagnostics/clean_screen_summary.csv", "results/clean_screen_summary.csv")
        for path in sorted((root / "evaluation/manifests").rglob("*.jsonl")):
            add_file(archive, path, f"manifests/{path.relative_to(root / 'evaluation/manifests')}")
        for path in sorted(examples_dir.iterdir()):
            add_file(archive, path, f"examples/{path.name}")
        for path in sorted((output / "figures").glob("*.png")):
            add_file(archive, path, f"figures/{path.name}")
    print(f"wrote archive={archive_path} bytes={archive_path.stat().st_size} examples={len(example_rows)}")


if __name__ == "__main__":
    main()
