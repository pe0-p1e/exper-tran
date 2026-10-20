"""CPU-only baseline preparation; an isolated, opt-in surrogate attack worker.

Optimizer arms are representation-loss adaptations, not paper reproductions.
Never changes the sources/locks of the ongoing original13 campaign.
"""
import argparse
import csv
import fcntl
import json
import sys
from pathlib import Path

from pp4_protocol import OUT, PREVIOUS, EXP, TIDS, condition, development_ids, sha, atomic_json, fingerprint
from supplement_registry import registry

ROOT = OUT / 'representation_baselines'
POOLS = {
    'Qwen3.5-27B': ['CLIP ViT-L/14', 'InternVL3.5-2B-HF', 'Gemma 4 E2B-it'],
    'InternVL3.5-14B-HF': ['CLIP ViT-L/14', 'Qwen3.5-2B', 'Gemma 4 E2B-it'],
    'Gemma 4 31B-it': ['CLIP ViT-L/14', 'Qwen3.5-2B', 'InternVL3.5-2B-HF'],
}
SOURCES = ['prepare_representation_baselines.py', 'pp4_objective.py', 'pp4_protocol.py',
           'pp4_worker.py', 'run_v7_attack.py', 'vision_features.py',
           'representation_objectives.py', 'attack_optimizers.py']


def plan():
    from family_protocol import require_excluded_family
    rows = []
    for target, models in POOLS.items():
        family = registry()[target]['family']
        require_excluded_family(models, family, registry())
        # MI is always enabled in the frozen runner. No false PGD/no-momentum label.
        arms = [('R_MI_deep', [1.], False, False),
                ('R_MI_four', [.25, .5, .75, 1.], False, False),
                ('R_DI_MI_four', [.25, .5, .75, 1.], True, False),
                ('R_TI_MI_four', [.25, .5, .75, 1.], False, True),
                ('R_DI_TI_MI_four', [.25, .5, .75, 1.], True, True)]
        for label, layers, di, ti in arms:
            rows.append(make_row(target, models, label, layers, di, ti))
        for model in models:
            rows.append(make_row(target, [model], 'matched_single', [.25, .5, .75, 1.], True, False))
    return rows


def make_row(target, models, label, layers, di, ti):
    c = condition(models, pull=.5, layers=layers, di=di)
    c['options'].update(smoothing=ti, excluded_proxy_family=registry()[target]['family'],
                        baseline_worker_sha256=sha(Path(__file__)))
    c.update(target=target, ti=ti)
    c['id'] = 'RBASE_' + fingerprint([target, models, c['options'], di])[:16]
    c['name'] = c['id'] + '__PP4__equal__' + '+'.join(n.replace('/', '_').replace(' ', '_') for n in models)
    if di:
        c['name'] += '__DI'
    if ti:
        c['name'] += '__TI'
    c['name'] += '__s42__i50__r1'
    return dict(label=label, condition=c)


def prepare():
    from commercial_matrix_pack import freeze
    rows = plan()
    lock = dict(protocol='representation_baseline_preparation_v1', status='PREPARED_NOT_RUN',
                conditions=rows, development_ids=development_ids(),
                model_revisions=registry(), source_hashes={f: sha(EXP / 'src' / f) for f in SOURCES},
                parameter_dtype='bfloat16', quantization=False, loss='cosine_pull_push',
                pull=.5, push=.5, CLS=0, patch=0, steps=50, seed=42,
                epsilon='16/255', step_size='1/255', momentum=1, random_start=True,
                fusion='equal per-image L1 normalized proxy gradients', reference_count=48,
                TI='5x5 Gaussian sigma1 zero-padded smoothing after fusion, before momentum',
                development_only=True, final_target_inference=False, commercial_testing=False,
                caveat='Previously inspected development10; not an independent holdout. '
                       'Validator families may overlap attack families; exact checkpoint exclusion only.',
                external_FOA_status='SOURCE_REVIEW_ONLY_NOT_IMPLEMENTED_OR_RUN')
    freeze(ROOT / 'protocol_lock.json', lock)
    for row in rows:
        c = row['condition']
        atomic_json(ROOT / 'configs' / (c['id'] + '.json'), c['options'])
    with (ROOT / 'matrix.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'label', 'evaluation_target_family_excluded',
            'models', 'layers', 'DI', 'TI', 'N_dev', 'status'])
        writer.writeheader()
        for row in rows:
            c = row['condition']
            writer.writerow(dict(id=c['id'], label=row['label'], evaluation_target_family_excluded=c['target'],
                models=' + '.join(c['models']), layers=c['options']['layers'], DI=c['di'], TI=c['ti'],
                N_dev=10, status='PREPARED_NOT_RUN'))
    # Link the running, hash-validated full comparison instead of making a duplicate queue.
    from pp4_original13_protocol import plan as original_plan, evidence
    inventory = []
    for row in original_plan():
        e = evidence(row['condition'], deep=True)
        inventory.append(dict(ensemble=row['id'], target=row['target'], attack_images=e['attack_images'],
            evaluated_images=len(e['measured']), baseline_new_PP=e['evaluation_complete'],
            source=str(OUT / 'pull_push_original13'), reuse='NO_NEW_RUN',
            historical_A3='method-level baseline; not coefficient-only ablation',
            historical_V6='historical baseline; layers, DI and coefficients differ'))
    atomic_json(ROOT / 'existing_original13_inventory.json', inventory)
    print(json.dumps(dict(status='PREPARED_NOT_RUN', conditions=len(rows), dev_images_per_condition=10,
                          GPU_models_loaded=0, output=str(ROOT))))
    return rows


def attack(cid, smoke=False):
    # Shares the active campaign's ownership lock; never competes with it.
    with (PREVIOUS / 'scheduler.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('GPU campaign owns scheduler.lock; baseline remains prepared, not started')
        lock = json.loads((ROOT / 'protocol_lock.json').read_text())
        for name, expected in lock['source_hashes'].items():
            if sha(EXP / 'src' / name) != expected:
                raise RuntimeError('Baseline scientific source changed: ' + name)
        c = next(r['condition'] for r in lock['conditions'] if r['condition']['id'] == cid)
        import pp4_worker as base
        import run_v7_attack as runner
        from pp4_objective import objective
        base.ROOT = ROOT
        base.install_audit()
        runner.objective = objective
        runner.METHODS['PP4'] = dict(c['options'])
        runner.prepare_data = base.shared.verify_existing_data
        original = runner.records
        def selected(path):
            data = original(path)
            return data[2:4] if Path(path).parent == OUT / 'manifests/dev' else data
        runner.records = selected
        sys.argv = [sys.argv[0], '--models', *c['models'], '--method', 'PP4', '--fusion', 'equal',
            '--split', 'dev', '--steps', '2' if smoke else '50', '--seed', '42', '--restarts', '1',
            '--batch-size', '2', '--limit-per-direction', '2', '--run-label', c['id'] + ('_SMOKE' if smoke else ''),
            '--excluded-proxy-family', c['options']['excluded_proxy_family'],
            '--config-json', str(ROOT / 'configs' / (c['id'] + '.json'))]
        if c['di']:
            sys.argv.append('--input-diversity')
        if c['ti']:
            sys.argv.append('--smoothing')
        if smoke:
            sys.argv += ['--transitions', TIDS[0]]
        runner.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', help='Explicit opt-in surrogate attack; does not evaluate target models')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and not args.condition:
        parser.error('--smoke requires --condition')
    if args.condition:
        attack(args.condition, args.smoke)
    else:
        prepare()
