import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_linear_comparison import make_arms, select_arm


def test_search_includes_cls_and_pull_only_and_infonce():
    plan = {"pairs": {"P16": {"rho": 1., "tau": .1, "push": .25}},
            "steps": 50, "step_size": 1/255,
            "rho_multipliers": [.5, 1., 2.], "push_weights": [0., .25, .5, 1.]}
    arms = make_arms(plan, "P16")
    assert len(arms) == 14
    assert len({a["name"] for a in arms}) == 14
    assert arms[0]["rho"] == 0
    assert arms[1]["semantic_mode"] == "prototype"
    assert sum(a["source_logit_weight"] == 0 for a in arms[2:]) == 3


def test_selection_uses_tuning_only_and_rejects_partial(tmp_path):
    arms = [dict(name="linear_a", rho=1., source_logit_weight=.5),
            dict(name="linear_b", rho=.5, source_logit_weight=.25)]
    for arm, hits in zip(arms, (3, 4)):
        directory = tmp_path / "states/P16/T02"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{arm['name']}.json").write_text(json.dumps(
            dict(status="complete", clean_valid_count=8, tasr_hits=hits)))
    assert select_arm(tmp_path, "P16", arms, ["T02"])["name"] == "linear_b"
    (directory / "linear_b.json").write_text(json.dumps(
        dict(status="complete", clean_valid_count=7, tasr_hits=7)))
    with pytest.raises(RuntimeError, match="N=8"):
        select_arm(tmp_path, "P16", arms, ["T02"])
