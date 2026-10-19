import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dynamic_pool import fits,estimate


def test_admission_reserves_future_peaks_and_limits_total_workers():
    assert fits(10.5,[8,9],12)
    assert not fits(10.5,[8,9,10],12)
    assert not fits(10.5,[8,9],60)
    assert fits(10.5,[],61)


def test_estimate_accounts_for_full_gemma_load_before_vision_only_execution():
    from v7_common import REGISTRY
    def event(name,size,vision):
        return dict(revision=REGISTRY[name]['revision'],parameter_bytes_by_dtype={'bfloat16':size*2**30},vision_encoder_parameter_bytes_by_dtype={'bfloat16':vision*2**30})
    names=['Gemma 4 E2B-it','Gemma 4 26B-A4B-it']
    events={names[0]:event(names[0],9.5,.3),names[1]:event(names[1],48,1.1)}
    reserve=estimate(names,'attack',events,{})
    assert reserve>48 # CPU offload after loading cannot erase the initial load peak.
    assert estimate([names[1]],'evaluation',events,{})>50


def test_orphan_recovery_cannot_adopt_reused_unrelated_pid(tmp_path,monkeypatch):
    import dynamic_pool
    import pytest
    monkeypatch.setattr(dynamic_pool,'alive',lambda pid:True)
    process=tmp_path/'99';process.mkdir();(process/'cmdline').write_text('unrelated_program')
    original=dynamic_pool.Path
    monkeypatch.setattr(dynamic_pool,'Path',lambda p:tmp_path if p=='/proc' else original(p))
    with pytest.raises(RuntimeError,match='another process'):
        dynamic_pool.recover_jobs([dict(pid=99,condition='A')],dict(A=dict(item={})))


def test_exclusive_target_memory_request_does_not_deadlock_shared_attack_queue():
    from dynamic_pool import evaluation_can_share
    assert not evaluation_can_share(10.5,64)
    assert evaluation_can_share(10.5,60)
    assert evaluation_can_share(0,64)
