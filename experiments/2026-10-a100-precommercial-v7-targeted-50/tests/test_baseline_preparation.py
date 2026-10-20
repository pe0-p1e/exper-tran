from prepare_representation_baselines import plan, POOLS
from supplement_registry import registry


def test_baselines_are_matched_and_family_disjoint():
    rows = plan()
    assert len(rows) == 24
    assert len({r['condition']['id'] for r in rows}) == 24
    reg = registry()
    for target, models in POOLS.items():
        group = [r for r in rows if r['condition']['target'] == target]
        singles = [r['condition']['models'][0] for r in group if r['label'] == 'matched_single']
        assert singles == models
        arms = {r['label']: r['condition'] for r in group if r['label'] != 'matched_single'}
        assert [(arms[k]['di'], arms[k]['ti']) for k in
                ['R_MI_four', 'R_DI_MI_four', 'R_TI_MI_four', 'R_DI_TI_MI_four']] == [
                    (False, False), (True, False), (False, True), (True, True)]
        assert arms['R_MI_deep']['options']['layers'] == [1.]
        for row in group:
            c = row['condition']
            assert c['split'] == 'dev' and c['steps'] == 50 and c['seed'] == 42
            assert all(reg[m]['family'] != reg[target]['family'] for m in c['models'])
            assert c['options']['loss'] == 'pull_push'
            assert c['options']['pull_weight'] == c['options']['push_weight'] == .5
            assert c['options']['patch_weight'] == c['options']['projector_weight'] == 0
