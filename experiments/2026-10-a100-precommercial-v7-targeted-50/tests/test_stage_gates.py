import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_proxy_selection as selection
import run_category_generalization as categories


def test_selection_requires_actual_finished_recovery_and_precision_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(selection,'OUT',tmp_path)
    with pytest.raises(RuntimeError):selection.require_recovery_complete()
    p=tmp_path/'family_disjoint_recovery';(p/'audits').mkdir(parents=True)
    (p/'state.json').write_text(json.dumps(dict(finished=1)))
    (p/'audits/validation.json').write_text(json.dumps(dict(all_attacks_validated=True,all_loaded_models_bf16=False)))
    with pytest.raises(RuntimeError):selection.require_recovery_complete()
    (p/'audits/validation.json').write_text(json.dumps(dict(all_attacks_validated=True,all_loaded_models_bf16=True)))
    selection.require_recovery_complete()


def test_category_generation_cannot_skip_proxy_measurements(tmp_path,monkeypatch):
    monkeypatch.setattr(categories,'OUT',tmp_path)
    with pytest.raises(RuntimeError):categories.require_selection_complete()
    p=tmp_path/'proxy_selection';p.mkdir()
    (p/'completion.json').write_text(json.dumps(dict(all_complete=True,feature_complete=False)))
    with pytest.raises(RuntimeError):categories.require_selection_complete()
    (p/'completion.json').write_text(json.dumps(dict(all_complete=True,feature_complete=True)))
    categories.require_selection_complete()
