"""No-model regression tests for frozen artifact identity and layer indexing."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vision_features import layer_indices
import run_v7_attack as runner
from v7_common import sha


def test_normalized_depth_matches_real_block_contract():
    assert layer_indices(24,[.01,.25,.5,.75,1.])==[0,5,11,17,23]
    assert layer_indices(16,[.01,.25,.5,.75,1.])==[0,3,7,11,15]


def test_resume_rejects_modified_or_missing_pngs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'OUT',tmp_path)
    png=tmp_path/'adv.png';png.write_bytes(b'original')
    state=tmp_path/'state.json'
    state.write_text(json.dumps(dict(status='attack_complete',identity='locked',png_hashes={'adv.png':sha(png)})))
    assert runner.valid_state(state,'locked')
    assert runner.valid_state(state,'different-method') is None
    png.write_bytes(b'tampered')
    assert runner.valid_state(state,'locked') is None
    png.unlink()
    assert runner.valid_state(state,'locked') is None
