"""Protect the concurrent owner's GPU budget and the predeclared contrast design."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_proxy_study import eligible, args_for
from proxy_study_protocol import plan


def test_admission_accounts_for_other_workers_and_primary():
    assert eligible(10.5, [15, 15], 20, 3)
    assert not eligible(10.5, [15, 15], 32, 3)
    assert not eligible(10.5, [8, 8, 8], 3, 4)
    assert eligible(0, [20, 20], 32, 2)


def test_plan_and_same_count_contrasts_exclude_target_family():
    rows = plan()
    assert len(rows) == 96
    assert len({(r['item']['config']['target'], frozenset(r['item']['config']['models'])) for r in rows}) == 96
    for target in {r['item']['config']['target'] for r in rows}:
        by = {r['template']: r['item']['config'] for r in rows if r['item']['config']['target'] == target}
        assert all(len(by[n]['models']) == 3 for n in ['M07', 'M16', 'M17'])
        assert all(len(by[n]['models']) == 4 for n in ['M15', 'M18', 'M23'])


def test_new_target_evaluation_uses_study_lock_and_frozen_test_pngs():
    row = next(r for r in plan() if r['item']['source'] == 'study18')
    args = list(map(str, args_for(row, 'evaluation', [])))
    assert args[0] == 'evaluation'
    assert '/attacks/test/' in args[args.index('--root')+1]
    assert '/proxy_composition_18h/method_lock.json' in args[args.index('--method-lock')+1]


def test_report_preserves_one_row_when_directed_metrics_already_include_target(monkeypatch, tmp_path):
    import time
    import proxy_study_protocol as protocol
    row=plan()[0]
    result=dict(evaluation_complete=True,attack_complete=True,attack_images=150,images=150,
                rows=[dict(target=row['item']['config']['target'],transition='T01',hits=20,denominator=30)],
                attack_seconds=100,peak_reserved_gib=10,mean_gradient_cosine=.2,
                negative_cosine_fraction=.1,mean_cancellation=.7)
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    monkeypatch.setattr(protocol,'checked',lambda *args,**kwargs:result)
    table=protocol.report([row],dict(started=time.time(),blocked={}))
    assert len(table)==1
    assert table[0]['status']=='COMPLETE'
    assert (tmp_path/'reports/directed_results.csv').exists()
