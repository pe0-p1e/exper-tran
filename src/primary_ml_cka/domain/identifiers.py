from dataclasses import dataclass
from enum import StrEnum


class ExperimentType(StrEnum):
    CROSS_FAMILY = "Cross-Family"
    INTRA_FAMILY = "Intra-Family"


@dataclass(frozen=True, slots=True)
class ModelPair:
    pair_id: str
    exp_type: ExperimentType
    proxy_model: str
    target_model: str


MODEL_PAIRS = (
    ModelPair("P02", ExperimentType.CROSS_FAMILY, "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it"),
    ModelPair(
        "P06",
        ExperimentType.CROSS_FAMILY,
        "openai/clip-vit-large-patch14",
        "OpenGVLab/InternVL3_5-2B-HF",
    ),
    ModelPair(
        "P11",
        ExperimentType.CROSS_FAMILY,
        "google/siglip2-so400m-patch14-384",
        "google/gemma-4-E2B-it",
    ),
    ModelPair("P14", ExperimentType.INTRA_FAMILY, "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"),
    ModelPair("P23", ExperimentType.CROSS_FAMILY, "Qwen/Qwen3.5-2B", "OpenGVLab/InternVL3_5-4B-HF"),
    ModelPair("P20", ExperimentType.INTRA_FAMILY, "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"),
    ModelPair(
        "P16",
        ExperimentType.INTRA_FAMILY,
        "OpenGVLab/InternVL3_5-2B-HF",
        "OpenGVLab/InternVL3_5-4B-HF",
    ),
    ModelPair(
        "P21",
        ExperimentType.INTRA_FAMILY,
        "OpenGVLab/InternVL3_5-4B-HF",
        "OpenGVLab/InternVL3_5-2B-HF",
    ),
    ModelPair(
        "P19",
        ExperimentType.INTRA_FAMILY,
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    ),
    ModelPair(
        "P22",
        ExperimentType.INTRA_FAMILY,
        "google/gemma-4-E4B-it",
        "google/gemma-4-E2B-it",
    ),
)

MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "OpenGVLab/InternVL3_5-2B-HF": "3f301ffcf3dcbb47893afae6650ea3e78d96fb6d",
    "OpenGVLab/InternVL3_5-4B-HF": "6bd4487402110ef9889ba50eb7aefeb302526fed",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "openai/clip-vit-base-patch32": "c7244be81152024ce0e99ac8d2e373a8953d9f9a",
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    # Immutable V6 campaign revisions. Kept here so all proxy/analysis taps
    # record the same model snapshot used by the runner.
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "OpenGVLab/InternVL3_5-8B-HF": "741a7d03020411e666c6109218ab71e08151ef86",
    "OpenGVLab/InternVL3_5-14B-HF": "226b96d5912e69159abc0384cefcbd51487fdce0",
    "google/gemma-4-26B-A4B-it": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "google/gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
}


def get_pair(pair_id: str) -> ModelPair:
    try:
        return next(pair for pair in MODEL_PAIRS if pair.pair_id == pair_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown pair ID: {pair_id}") from exc
