"""Shared configuration helpers for the self-contained V4 experiment."""

import math
from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, get_pair
from primary_ml_cka.models.common.loading import local_snapshot

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "a100_final.yaml"
DEFAULT_OUTPUT = Path("outputs/a100_final_v5")


@dataclass(frozen=True, slots=True)
class Transition:
    transition_id: str
    source: int
    target: int


def load_experiment(path: Path = DEFAULT_CONFIG) -> dict:
    raw = load_config(path)
    validate_experiment(raw)
    return raw


def transitions(raw: dict) -> tuple[Transition, ...]:
    return tuple(
        Transition(str(item["id"]), int(item["source"]), int(item["target"]))
        for item in raw["transitions"]
    )


def pair_specs(raw: dict) -> dict[str, dict]:
    return {str(item["pair_id"]): item for item in raw["pairs"]}


def class_specs(raw: dict) -> tuple[dict, ...]:
    return tuple(sorted(raw["classes"], key=lambda item: int(item["label"])))


def class_names(raw: dict) -> tuple[str, ...]:
    return tuple(str(item["name"]) for item in class_specs(raw))


def vision_layer_count(model_id: str, hf_home: Path = Path(".hf-cache")) -> int:
    """Read vision depth from the pinned local model config."""
    from transformers import AutoConfig

    snapshot = local_snapshot(hf_home, model_id, MODEL_REVISIONS[model_id])
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    candidates = [config, getattr(config, "vision_config", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for field in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(candidate, field, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError(f"Cannot discover vision layer count for {model_id}")


def layer_from_percent(model_id: str, percent: float, hf_home: Path = Path(".hf-cache")) -> int:
    if not 0 < percent <= 100:
        raise ValueError("representation layer percentage must be in (0, 100]")
    count = vision_layer_count(model_id, hf_home)
    return min(count - 1, max(0, math.ceil(count * percent / 100) - 1))


def classification_prompt(raw: dict) -> str:
    options = "; ".join(f"{index} {name}" for index, name in enumerate(class_names(raw)))
    return (
        "Classify the main object in the image into exactly one of the following "
        f"categories:\n{options}.\n"
        "Return only one integer from 0 to 9.\n"
        "Do not output reasoning, punctuation, or additional words."
    )


def validate_experiment(raw: dict) -> None:
    classes = class_specs(raw)
    labels = [int(item["label"]) for item in classes]
    wnids = [str(item["wnid"]) for item in classes]
    names = [str(item["name"]) for item in classes]
    domains = [str(item["domain"]) for item in classes]
    if labels != list(range(1, 11)):
        raise ValueError("Diverse class labels must be exactly 1..10")
    if len(set(wnids)) != 10 or len(set(names)) != 10:
        raise ValueError("Diverse class WNIDs and names must be unique")
    if len(set(domains)) < 8:
        raise ValueError("The catalog must span at least eight semantic domains")
    candidate_split = str(raw.get("candidate_split", "val"))
    candidate_offset = int(raw.get("candidate_offset", 0))
    if candidate_split not in {"train", "val"}:
        raise ValueError("candidate_split must be train or val")
    if candidate_offset < 0:
        raise ValueError("candidate_offset must be non-negative")
    if candidate_split == "train" and candidate_offset < int(raw["reference_count"]):
        raise ValueError("Train candidates must start after the reference bank")
    items = tuple(raw["transitions"])
    catalog_mode = str(raw.get("catalog_mode", "diverse10"))
    if catalog_mode == "three_class_transfer":
        expected = {
            (1, 2),
            (1, 7),
            (2, 1),
            (2, 7),
            (7, 1),
            (7, 2),
        }
        actual = {(int(item["source"]), int(item["target"])) for item in items}
        if actual != expected or len(items) != 6:
            raise ValueError(
                "three_class_transfer requires the six directed edges among labels 1, 2, and 7"
            )
    elif catalog_mode == "full_directed_pairs":
        expected_edges = {
            (source, target)
            for source in range(1, 11)
            for target in range(1, 11)
            if source != target
        }
        actual_edges = {
            (int(item["source"]), int(item["target"])) for item in items
        }
        if len(items) != 90 or actual_edges != expected_edges:
            raise ValueError("full_directed_pairs requires all 90 non-self directed edges")
    elif len(items) != 10:
        raise ValueError("V4 requires exactly ten transitions")
    ids = [str(item["id"]) for item in items]
    sources = [int(item["source"]) for item in items]
    targets = [int(item["target"]) for item in items]
    expected_count = {
        "three_class_transfer": 6,
        "full_directed_pairs": 90,
    }.get(catalog_mode, 10)
    if len(set(ids)) != expected_count:
        raise ValueError("Transition IDs must be unique")
    if catalog_mode not in {"three_class_transfer", "full_directed_pairs"} and sorted(sources) != list(range(1, 11)):
        raise ValueError("Every class must appear exactly once as a source")
    if catalog_mode not in {"three_class_transfer", "full_directed_pairs"} and sorted(targets) != list(range(1, 11)):
        raise ValueError("Every class must appear exactly once as a target")
    if any(source == target for source, target in zip(sources, targets, strict=True)):
        raise ValueError("Source and target must differ")
    edges = {frozenset((source, target)) for source, target in zip(sources, targets, strict=True)}
    if catalog_mode == "directed_pairs":
        directed = set(zip(sources, targets, strict=True))
        if len(edges) != 5 or any((target, source) not in directed for source, target in directed):
            raise ValueError(
                "directed_pairs requires five class pairs, each represented in both directions"
            )
    elif catalog_mode not in {"three_class_transfer", "full_directed_pairs"} and len(edges) != 10:
        raise ValueError("Undirected transition edges must not repeat")
    for spec in raw["pairs"]:
        pair = get_pair(str(spec["pair_id"]))
        if pair.pair_id not in {"P02", "P14", "P16", "P19", "P20", "P21", "P22", "P23"}:
            raise ValueError("A100 final run uses the pinned generative proxy pair matrix")
    arms = {str(arm["name"]): arm for arm in raw["arms"]}
    required = (
        {"full_pull_push", "second_loss_only"}
        if catalog_mode == "three_class_transfer"
        else {
            "pull_push_standard",
            "multiclass_standard",
            "pull_push_small_steps",
            "multiclass_small_steps",
        }
    )
    if set(arms) != required:
        raise ValueError(
            "The configured controlled loss arms are incomplete or unexpected: "
            f"expected {sorted(required)}"
        )
    for arm in arms.values():
        if int(arm["steps"]) < 1 or float(arm["step_size"]) <= 0:
            raise ValueError("Attack steps and step size must be positive")


def transition_dir(output_dir: Path, transition_id: str) -> Path:
    return output_dir / "evaluation" / "manifests" / "transitions" / transition_id
