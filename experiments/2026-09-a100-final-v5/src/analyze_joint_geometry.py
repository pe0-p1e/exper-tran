#!/usr/bin/env python3
"""Joint target-space PCA/t-SNE, class variance, and transfer asymmetry analysis."""

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from common import class_names, load_experiment, transitions
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from primary_ml_cka.artifacts.png import load_png_tensor
from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.data.preprocessing import ensure_canvas
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.models.analysis.representations import extract_representations
from primary_ml_cka.models.proxies.registry import load_proxy


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dispersion(values: torch.Tensor) -> dict:
    z = functional.normalize(values.float(), dim=-1)
    mean = z.mean(0)
    center = functional.normalize(mean, dim=0)
    per_image = 1 - z @ center
    centered = z - mean
    covariance_trace = centered.square().sum() / max(1, len(z) - 1)
    eigenvalues = torch.linalg.svdvals(centered).square()
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "center": center,
        "dispersion": float(per_image.mean()),
        "dispersion_std": float(per_image.std(unbiased=False)),
        "covariance_trace": float(covariance_trace),
        "effective_rank": float(effective_rank),
    }


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.float() - left.float().mean(dim=0, keepdim=True)
    y = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = x.T @ y
    denominator = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    return float(cross.square().sum() / denominator.clamp_min(1e-12))


def rsa(left: torch.Tensor, right: torch.Tensor) -> float:
    x = functional.normalize(left.float(), dim=-1)
    y = functional.normalize(right.float(), dim=-1)
    indices = torch.triu_indices(len(x), len(x), offset=1)
    return float(
        spearmanr(
            (x @ x.T)[indices[0], indices[1]].numpy(), (y @ y.T)[indices[0], indices[1]].numpy()
        ).statistic
    )


def load_complete_transition(
    output_dir: Path, raw: dict, pair_id: str, transition, *, allow_attack_complete: bool = False
):
    state_dir = (
        output_dir
        / str(raw.get("state_namespace", "states_a100_v5"))
        / pair_id
        / transition.transition_id
    )
    states = [json.loads(path.read_text()) for path in sorted(state_dir.glob("batch_*.json"))]
    expected = math.ceil(int(raw["attack_count"]) / int(raw.get("batch_size", 32)))
    allowed = {"complete", "attack_complete"} if allow_attack_complete else {"complete"}
    if len(states) != expected or any(s.get("status") not in allowed for s in states):
        return None
    return states


def plot_joint_directions(
    *, raw: dict, pair_id: str, plot_transition: str, output_dir: Path,
    state_map: dict, ref_records: list, indexed: dict, row_of: dict, analysis: Path,
) -> None:
    """Fit PCA/t-SNE once on both class clouds and both attack directions."""
    transition = next(
        (item for item in transitions(raw) if item.transition_id == plot_transition), None
    )
    if transition is None or transition.transition_id not in state_map:
        return
    names = class_names(raw)
    selected = []
    labels = []
    for label in (transition.source, transition.target):
        refs = [
            output_dir / "canonical_images" / record.relative_path
            for current, record in ref_records
            if current == label
        ]
        selected.extend(row_of[path] for path in refs)
        labels.extend([f"reference:{names[label-1]}"] * len(refs))
    reverse = next(
        (
            item for item in transitions(raw)
            if item.source == transition.target
            and item.target == transition.source
            and item.transition_id in state_map
        ),
        None,
    )
    plotted = {transition.transition_id}
    if reverse is not None:
        plotted.add(reverse.transition_id)
    transition_by_id = {item.transition_id: item for item in transitions(raw)}
    for (tid, _batch, _index), (clean_path, adv_path) in indexed.items():
        if tid in plotted:
            direction = transition_by_id[tid]
            selected.extend((row_of[clean_path], row_of[adv_path]))
            labels.extend(
                (
                    f"clean:{names[direction.source-1]}",
                    f"adv:{names[direction.source-1]}→{names[direction.target-1]}",
                )
            )
    x = functional.normalize(torch.stack(selected).float(), dim=-1).numpy()
    pca_xy = PCA(n_components=2, random_state=42).fit_transform(x)
    perplexity = min(30.0, max(2.0, (len(x) - 1) / 3))
    pre = PCA(n_components=min(50, x.shape[1], x.shape[0] - 1), random_state=42).fit_transform(x)
    tsne_xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    ).fit_transform(pre)
    import matplotlib.pyplot as plt

    for name, coords in (("pca", pca_xy), ("tsne", tsne_xy)):
        fig, ax = plt.subplots(figsize=(10, 8))
        for group in dict.fromkeys(labels):
            indices = [i for i, value in enumerate(labels) if value == group]
            ax.scatter(coords[indices, 0], coords[indices, 1], s=20, alpha=0.7, label=group)
        if name == "pca":
            clean_indices = [i for i, value in enumerate(labels) if value.startswith("clean:")]
            adv_indices = [i for i, value in enumerate(labels) if value.startswith("adv:")]
            for clean_index, adv_index in zip(clean_indices, adv_indices, strict=False):
                ax.annotate(
                    "",
                    xy=coords[adv_index],
                    xytext=coords[clean_index],
                    arrowprops={"arrowstyle": "->", "alpha": 0.18, "lw": 0.7},
                )
        ax.set_title(
            f"Joint {name.upper()}: {names[transition.source-1]} ↔ {names[transition.target-1]}"
        )
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = analysis / name / f"{pair_id}_{plot_transition}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/a100_final.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/a100_final_v5"))
    parser.add_argument("--pair", default="P23")
    parser.add_argument("--plot-transition", default="T01")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hf-home", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--analysis-subdir", default="", help="Optional output subdirectory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Measure frozen attack PNGs even when target classification or some transitions are unfinished",
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Cache full-90 embeddings and draw joint plots before target labels finish",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for target representation extraction")
    raw = load_experiment(args.config)
    if args.batch_size is not None:
        raw["batch_size"] = args.batch_size
    output_dir = args.output_dir.resolve()
    pair = get_pair(args.pair)
    ref_records = []
    for label in range(1, 11):
        records = read_manifest(
            output_dir / "evaluation/manifests" / f"class_references_{label:02d}.jsonl"
        )
        ref_records.extend((label, r) for r in records)
    paths: list[Path] = [output_dir / "canonical_images" / r.relative_path for _, r in ref_records]
    indexed = {}
    state_map = {}
    artifact_root = output_dir / "attacks" / args.pair
    for transition in transitions(raw):
        states = load_complete_transition(
            output_dir, raw, args.pair, transition,
            allow_attack_complete=args.precompute_only or args.allow_partial,
        )
        if states is None:
            continue
        state_map[transition.transition_id] = states
        for state in states:
            batch = int(state["batch_index"])
            directory = (
                artifact_root
                / f"a100_v5_{transition.transition_id}"
                / f"batch_{batch:02d}"
                / str(state["objective_tag"])
                / "lambda_1"
            )
            for i in range(int(state["source_count"])):
                key = (transition.transition_id, batch, i)
                clean_path, adv_path = (
                    directory / f"{i:02d}_clean.png",
                    directory / f"{i:02d}_adv.png",
                )
                if not clean_path.is_file() or not adv_path.is_file():
                    raise FileNotFoundError(f"Missing frozen PNG pair for {key}")
                indexed[key] = (clean_path, adv_path)
                paths.extend((clean_path, adv_path))
    if not state_map:
        raise RuntimeError("No complete A100 v5 transitions are available")
    if args.precompute_only and not args.allow_partial and len(state_map) != len(transitions(raw)):
        raise RuntimeError(
            f"Precompute requires all frozen attack PNGs: {len(state_map)}/{len(transitions(raw))}"
        )
    paths = list(dict.fromkeys(paths))
    fingerprint = hashlib.sha256("\n".join(str(p.resolve()) for p in paths).encode()).hexdigest()

    def features_for(model_id: str) -> torch.Tensor:
        key = model_id.replace("/", "_")
        cache = output_dir / "analysis/embeddings" / f"{args.pair}_{key}_{fingerprint[:16]}.pt"
        if args.resume and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            if payload.get("path_fingerprint") == fingerprint:
                return payload["features"]
        result = extract_representations(
            model_id, paths, args.hf_home, torch.device("cuda"), precision="bf16"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": result, "path_fingerprint": fingerprint}, cache)
        return result

    features = features_for(pair.target_model)
    selected_layers = {
        int(state["representation_layer"])
        for states in state_map.values()
        for state in states
    }
    if len(selected_layers) != 1:
        raise RuntimeError(
            f"Layer analysis received mixed representation layers: {selected_layers}"
        )
    selected_layer = selected_layers.pop()
    proxy_key = pair.proxy_model.replace("/", "_")
    proxy_cache = (
        output_dir
        / "analysis/embeddings"
        / f"{args.pair}_{proxy_key}_layer_{selected_layer}_{fingerprint[:16]}.pt"
    )
    if args.resume and proxy_cache.is_file():
        payload = torch.load(proxy_cache, map_location="cpu", weights_only=True)
        proxy_features = payload["features"]
    else:
        attack_config = AttackConfig(generative_precision="bf16")
        proxy_model = load_proxy(
            pair.proxy_model, args.hf_home, torch.device("cuda"), attack_config
        )
        proxy_rows = []
        try:
            with torch.no_grad():
                for path in paths:
                    image = load_png_tensor(path).cuda().unsqueeze(0)
                    image = ensure_canvas(image, attack_config.canvas_size)
                    embedded = proxy_model.image_embeddings(
                        image,
                        representation_type="vision_encoder",
                        layer=selected_layer,
                        pooling="mean",
                    )
                    vector = (
                        embedded.semantic_embeddings
                        if embedded.semantic_embeddings is not None
                        else embedded.embeddings
                    )
                    proxy_rows.append(vector[0].float().cpu())
        finally:
            del proxy_model
            gc.collect()
            torch.cuda.empty_cache()
        proxy_features = torch.stack(proxy_rows)
        proxy_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": proxy_features, "path_fingerprint": fingerprint}, proxy_cache)
    row_of = {path: features[i] for i, path in enumerate(paths)}
    proxy_row_of = {path: proxy_features[i] for i, path in enumerate(paths)}
    subdir = Path(args.analysis_subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("analysis subdirectory must stay inside analysis/")
    analysis = output_dir / "analysis" / subdir
    if args.precompute_only:
        candidates = {}
        for candidate in transitions(raw):
            if candidate.transition_id in state_map:
                key = tuple(sorted((candidate.source, candidate.target)))
                if key not in candidates or candidate.source < candidate.target:
                    candidates[key] = candidate
        for candidate in candidates.values():
            if args.resume and all(
                (analysis / kind / f"{args.pair}_{candidate.transition_id}.png").is_file()
                for kind in ("pca", "tsne")
            ):
                continue
            plot_joint_directions(
                raw=raw,
                pair_id=args.pair,
                plot_transition=candidate.transition_id,
                output_dir=output_dir,
                state_map=state_map,
                ref_records=ref_records,
                indexed=indexed,
                row_of=row_of,
                analysis=analysis,
            )
        print(
            f"Precomputed target/proxy embeddings and {len(candidates)} joint class-pair plots under {analysis} "
            f"(image paths={len(paths)})",
            flush=True,
        )
        return
    names = class_names(raw)
    class_stats = {}
    class_image_paths = {}
    variance_rows = []
    for label in range(1, 11):
        class_paths = [
            output_dir / "canonical_images" / r.relative_path
            for current, r in ref_records
            if current == label
        ]
        class_image_paths[label] = class_paths
        stats = dispersion(torch.stack([row_of[p] for p in class_paths]))
        class_stats[label] = stats
        variance_rows.append(
            {
                "pair_id": args.pair,
                "target_model": pair.target_model,
                "label": label,
                "class_name": names[label - 1],
                "reference_n": len(class_paths),
                **{k: v for k, v in stats.items() if k != "center"},
            }
        )
    write_csv(analysis / "class_variance.csv", variance_rows)

    metric_rows = []
    image_rows = []
    directed_cache = {}
    for transition in transitions(raw):
        states = state_map.get(transition.transition_id)
        if states is None:
            continue
        clean_values, adv_values, target_hits, proxy_hits = [], [], [], []
        for state in states:
            batch = int(state["batch_index"])
            hits = state.get("target", {}).get("target_hit_mask", [])
            proxy_mask = state["attack"].get("proxy_target_hit_mask", [])
            for i in range(int(state["source_count"])):
                clean_path, adv_path = indexed[(transition.transition_id, batch, i)]
                clean_values.append(row_of[clean_path])
                adv_values.append(row_of[adv_path])
                target_hits.append(bool(hits[i]) if i < len(hits) else None)
                proxy_hits.append(bool(proxy_mask[i]) if i < len(proxy_mask) else None)
        clean_z = functional.normalize(torch.stack(clean_values).float(), dim=-1)
        adv_z = functional.normalize(torch.stack(adv_values).float(), dim=-1)
        source_center = class_stats[transition.source]["center"]
        target_center = class_stats[transition.target]["center"]
        geometry_paths = class_image_paths[transition.source] + class_image_paths[transition.target]
        proxy_geometry = torch.stack([proxy_row_of[path] for path in geometry_paths])
        target_geometry = torch.stack([row_of[path] for path in geometry_paths])
        s0, t0 = clean_z @ source_center, clean_z @ target_center
        s1, t1 = adv_z @ source_center, adv_z @ target_center
        pull = t1 - t0
        push = s0 - s1
        delta_r = pull + push
        for index in range(len(clean_values)):
            image_rows.append(
                {
                    "pair_id": args.pair,
                    "transition_id": transition.transition_id,
                    "image_index": index,
                    "source_label": transition.source,
                    "target_label": transition.target,
                    "target_hit": target_hits[index],
                    "proxy_target_hit": proxy_hits[index],
                    "delta_pull": float(pull[index]),
                    "delta_push": float(push[index]),
                    "delta_R": float(delta_r[index]),
                    "clean_target_margin": float(t0[index] - s0[index]),
                    "adversarial_target_margin": float(t1[index] - s1[index]),
                    "gap_closure": float((t1[index] - s1[index]) - (t0[index] - s0[index])),
                }
            )
        row = {
            "pair_id": args.pair,
            "transition_id": transition.transition_id,
            "source_label": transition.source,
            "target_label": transition.target,
            "N": len(clean_values),
            "N_target_evaluated": sum(hit is not None for hit in target_hits),
            "N_proxy_success": sum(hit is True for hit in proxy_hits),
            "TASR": (
                sum(p is True and t is True for p, t in zip(proxy_hits, target_hits, strict=True))
                / sum(p is True for p in proxy_hits)
                if all(t is not None for t in target_hits) and sum(p is True for p in proxy_hits)
                else None
            ),
            "unconditional_TASR": (sum(hit is True for hit in target_hits) / len(target_hits)) if all(hit is not None for hit in target_hits) else None,
            "source_variance": class_stats[transition.source]["dispersion"],
            "target_variance": class_stats[transition.target]["dispersion"],
            "source_covariance_trace": class_stats[transition.source]["covariance_trace"],
            "target_covariance_trace": class_stats[transition.target]["covariance_trace"],
            "source_effective_rank": class_stats[transition.source]["effective_rank"],
            "target_effective_rank": class_stats[transition.target]["effective_rank"],
            "prototype_distance": float(1 - source_center @ target_center),
            "CKA": linear_cka(proxy_geometry, target_geometry),
            "RSA": rsa(proxy_geometry, target_geometry),
            "mean_delta_pull": float(pull.mean()),
            "mean_delta_push": float(push.mean()),
            "mean_delta_R": float(delta_r.mean()),
            "median_delta_R": float(delta_r.median()),
            "clean_target_margin": float((t0 - s0).mean()),
            "adversarial_target_margin": float((t1 - s1).mean()),
            "margin_change": float(((t1 - s1) - (t0 - s0)).mean()),
            "gap_closure": float(((t1 - s1) - (t0 - s0)).mean()),
        }
        metric_rows.append(row)
        directed_cache[(transition.source, transition.target)] = row
    write_csv(analysis / "representation_shift.csv", metric_rows)
    write_csv(analysis / "per_image_representation_shift.csv", image_rows)
    asymmetry_rows = []
    for (source, target), forward in directed_cache.items():
        backward = directed_cache.get((target, source))
        if backward:
            asymmetry_rows.append(
                {
                    "source_label": source,
                    "target_label": target,
                    "TASR_forward": forward["TASR"],
                    "TASR_reverse": backward["TASR"],
                    "delta_TASR": (forward["TASR"] - backward["TASR"]) if forward["TASR"] is not None and backward["TASR"] is not None else None,
                    "variance_difference": forward["source_variance"] - forward["target_variance"],
                    "clean_margin_difference": forward["clean_target_margin"]
                    - backward["clean_target_margin"],
                    "gap_closure_difference": forward["gap_closure"] - backward["gap_closure"],
                }
            )
    write_csv(analysis / "asymmetry.csv", asymmetry_rows)

    numeric = [
        key
        for key in metric_rows[0]
        if any(isinstance(row.get(key), int | float) for row in metric_rows)
        and key not in {"source_label", "target_label", "N", "N_target_evaluated", "N_proxy_success"}
    ]
    correlation_rows = []
    for left in numeric:
        for right in numeric:
            observed = [(r[left], r[right]) for r in metric_rows if r[left] is not None and r[right] is not None]
            value = spearmanr([a for a, _ in observed], [b for _, b in observed]).statistic if len(observed) > 1 else float("nan")
            correlation_rows.append(
                {"metric_x": left, "metric_y": right, "spearman_rho": value, "n": len(observed)}
            )
    write_csv(analysis / "correlation_matrix.csv", correlation_rows)
    import matplotlib.pyplot as plt

    matrix = np.array(
        [
            [
                next(
                    (
                        r["spearman_rho"]
                        for r in correlation_rows
                        if r["metric_x"] == x and r["metric_y"] == y
                    ),
                    np.nan,
                )
                for y in numeric
            ]
            for x in numeric
        ]
    )
    fig, ax = plt.subplots(figsize=(max(8, len(numeric) * 0.6), max(7, len(numeric) * 0.55)))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numeric)), numeric, rotation=75, ha="right")
    ax.set_yticks(range(len(numeric)), numeric)
    fig.colorbar(image, ax=ax, label="Spearman rho")
    fig.tight_layout()
    (analysis / "correlation_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(analysis / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    transition = next(
        (item for item in transitions(raw) if item.transition_id == args.plot_transition), None
    )
    if transition and transition.transition_id in state_map:
        selected = []
        labels = []
        for label in (transition.source, transition.target):
            refs = [
                output_dir / "canonical_images" / r.relative_path
                for current, r in ref_records
                if current == label
            ]
            selected.extend(row_of[p] for p in refs)
            labels.extend([f"reference:{names[label-1]}"] * len(refs))
        reverse = next(
            (
                item
                for item in transitions(raw)
                if item.source == transition.target
                and item.target == transition.source
                and item.transition_id in state_map
            ),
            None,
        )
        plotted = {transition.transition_id}
        if reverse is not None:
            plotted.add(reverse.transition_id)
        plotted_transitions = {item.transition_id: item for item in transitions(raw)}
        for (tid, _batch, _index), (clean_path, adv_path) in indexed.items():
            if tid in plotted:
                direction = plotted_transitions[tid]
                selected.extend((row_of[clean_path], row_of[adv_path]))
                labels.extend(
                    (
                        f"clean:{names[direction.source-1]}",
                        f"adv:{names[direction.source-1]}→{names[direction.target-1]}",
                    )
                )
        x = functional.normalize(torch.stack(selected).float(), dim=-1).numpy()
        pca = PCA(n_components=2, random_state=42)
        pca_xy = pca.fit_transform(x)
        perplexity = min(30.0, max(2.0, (len(x) - 1) / 3))
        pre = PCA(n_components=min(50, x.shape[1], x.shape[0] - 1), random_state=42).fit_transform(
            x
        )
        tsne_xy = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
        ).fit_transform(pre)
        import matplotlib.pyplot as plt

        for name, coords in (("pca", pca_xy), ("tsne", tsne_xy)):
            fig, ax = plt.subplots(figsize=(10, 8))
            for group in dict.fromkeys(labels):
                indices = [i for i, value in enumerate(labels) if value == group]
                ax.scatter(coords[indices, 0], coords[indices, 1], s=20, alpha=0.7, label=group)
            if name == "pca":
                clean_indices = [i for i, value in enumerate(labels) if value.startswith("clean:")]
                adv_indices = [i for i, value in enumerate(labels) if value.startswith("adv:")]
                for clean_index, adv_index in zip(clean_indices, adv_indices, strict=False):
                    ax.annotate(
                        "",
                        xy=coords[adv_index],
                        xytext=coords[clean_index],
                        arrowprops={"arrowstyle": "->", "alpha": 0.18, "lw": 0.7},
                    )
            ax.set_title(
                f"Joint {name.upper()}: {names[transition.source-1]} ↔ {names[transition.target-1]}"
            )
            ax.legend(fontsize="small")
            fig.tight_layout()
            path = analysis / name / f"{args.pair}_{transition.transition_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)
    for candidate in transitions(raw):
        if candidate.source < candidate.target and candidate.transition_id != args.plot_transition:
            if args.resume and all(
                (analysis / kind / f"{args.pair}_{candidate.transition_id}.png").is_file()
                for kind in ("pca", "tsne")
            ):
                continue
            plot_joint_directions(
                raw=raw,
                pair_id=args.pair,
                plot_transition=candidate.transition_id,
                output_dir=output_dir,
                state_map=state_map,
                ref_records=ref_records,
                indexed=indexed,
                row_of=row_of,
                analysis=analysis,
            )
    print(f"Wrote full-90 representation diagnostics and 45 joint class-pair plots under {analysis}")


if __name__ == "__main__":
    main()
