import csv
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import proxy_selection_tables as tables
from family_protocol import require_excluded_family
from v7_common import REGISTRY


def test_selection_matrix_covers_all_seven_subsets_without_family_overlap():
    rows=tables.matrix()
    assert len(rows)==21
    for target in {r['config']['target'] for r in rows}:
        selected=[r for r in rows if r['config']['target']==target]
        assert {r['slots'] for r in selected}=={'V','A','B','V+A','V+B','A+B','V+A+B'}
        for r in selected:
            c=r['config']
            require_excluded_family(c['models'],REGISTRY[target]['family'],REGISTRY)
            assert c['steps']==50 and c['options']=={'patch_weight':.35} and c['di']


def test_zero_hits_is_complete_but_missing_or_partial_denominator_is_not():
    assert tables.complete_cell(dict(hits='0',denominator='30',status='complete'))
    assert not tables.complete_cell({})
    assert not tables.complete_cell(dict(hits='0',denominator='2',status='complete'))
    with pytest.raises(ValueError):tables.complete_cell(dict(hits='3',denominator='2'))


def test_partial_measurements_do_not_become_headline_full_cohort_asr(tmp_path,monkeypatch):
    monkeypatch.setattr(tables,'OUT',tmp_path)
    source=tmp_path/'source.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','target','transition','hits','denominator','status']);w.writeheader()
        w.writerow(dict(condition='frozen',target='Qwen3.5-27B',transition='T01',hits=1,denominator=2,status='partial'))
        w.writerow(dict(condition='frozen',target='Qwen3.5-27B',transition='T03',hits=0,denominator=30,status='complete'))
    plan=[dict(config=dict(id='x',target='Qwen3.5-27B',models=['CLIP ViT-L/14']),slots='V',source_csv='source.csv',measurement_condition='frozen',source='test')]
    rows,directions,_=tables.collect(plan)
    assert rows[0]['denominator']==32 and rows[0]['hits']==1
    assert rows[0]['ASR'] is None and rows[0]['status']=='partial'
    assert next(r for r in directions if r['transition']=='T03')['ASR']==0
