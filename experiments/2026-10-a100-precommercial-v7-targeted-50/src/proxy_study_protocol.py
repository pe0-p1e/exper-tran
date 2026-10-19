"""Isolated 18-hour controlled composition study; no changes to frozen V7 code."""
import csv
import json
import os
from pathlib import Path

import v7_common

if os.environ.get('V7_DATA_ROOT'):
    v7_common.OUT = Path(os.environ['V7_DATA_ROOT']).resolve()
from v7_common import OUT, EXP, atomic_json, sha, fingerprint, TIDS, records
from supplement_registry import registry
from family_protocol import condition_name, require_excluded_family

ROOT = OUT / 'proxy_composition_18h'
PLAN = EXP / 'CONTROLLED_PROXY_COMPOSITION_PLAN.csv'
# Complete scientific contrasts before a larger saturation test. Fixed before launch.
ORDER = [11, 12, 16, 17, 8, 9, 10, 13, 14, 15, 18, 1, 2, 3, 4, 5, 6, 7,
         19, 20, 21, 22, 27, 28, 29, 30, 23, 24, 25, 26, 31, 32]


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError('Immutable study artifact changed: ' + str(path))
    else:
        atomic_json(path, value)


def plan():
    from supplement_protocol import candidates
    previous = {(x['config']['target'], frozenset(x['config']['models'])): x
                for x in candidates()}
    rows = []
    for row in csv.DictReader(PLAN.open()):
        names = row['models'].split(' + ')
        target = row['target']
        require_excluded_family(names, registry()[target]['family'], registry())
        if len(names) != len(set(names)):
            raise ValueError('Duplicate checkpoint')
        for family in ['qwen', 'internvl', 'gemma4']:
            if sum(registry()[n]['family'] == family for n in names) > 2:
                raise ValueError('VLM family cap')
        old = previous.get((target, frozenset(names)))
        # Match an existing row's actual execution order; DI/RNG is not assumed commutative.
        if old:
            item = old
        else:
            prefix = {'Qwen3.5-27B': 'Q', 'InternVL3.5-14B-HF': 'I', 'Gemma 4 31B-it': 'G'}[target]
            c = dict(id='P18_' + row['id'] + '_' + prefix, target=target, models=names,
                     method='A3', options={'patch_weight': .35}, di=True, fusion='equal',
                     steps=50, seed=42, restarts=1, kind='controlled_composition')
            item = dict(config=c, measurement_condition=condition_name(c), source='study18',
                        key=fingerprint([target, names]), aliases=[row['id']], optional=False)
        rows.append(dict(template=row['id'], tier=row['tier'], item=item))
    rows.sort(key=lambda r: (ORDER.index(int(r['template'][1:])), r['item']['config']['target']))
    return rows


def checked(item, deep=False):
    import supplement_protocol as old
    saved = old.ROOT
    try:
        if item['source'] == 'study18':
            old.ROOT = ROOT
        return old.evidence(item, deep=deep)
    finally:
        old.ROOT = saved


def evaluator_hashes():
    repo = EXP.parents[1]
    files = [EXP / 'src/evaluate_v7.py',
             repo / 'src/primary_ml_cka/models/targets/generation.py',
             repo / 'src/primary_ml_cka/prompts/parser.py',
             repo / 'src/primary_ml_cka/models/backends/target_transformers_generation.py',
             repo / 'experiments/2026-09-a100-final-v5/src/common.py',
             repo / 'experiments/2026-10-a100-precommercial-v6/config/runner_template.yaml']
    return {str(p.relative_to(repo)): sha(p) for p in files}


def lock(rows):
    configs = [r['item']['config'] for r in rows if r['item']['source'] == 'study18']
    payload = dict(protocol='strict_family_disjoint', configs=configs,
                   allowed_test_conditions=[condition_name(c) for c in configs],
                   condition_targets={condition_name(c): c['target'] for c in configs},
                   condition_models={condition_name(c): c['models'] for c in configs},
                   model_revisions=registry(),
                   manifest_hashes={t: sha(OUT / 'manifests/test' / (t + '.jsonl')) for t in TIDS},
                   reference_hashes={str(i): sha(OUT / 'manifests/class_references' / f'class_references_{i:02d}.jsonl') for i in range(1, 11)},
                   source_hashes={n: sha(EXP / 'src' / n) for n in
                                  ['run_v7_attack.py', 'representation_objectives.py',
                                   'attack_optimizers.py', 'vision_features.py',
                                   'proxy_study_protocol.py', 'proxy_study_worker.py',
                                   'run_proxy_study.py', 'supplement_registry.py']},
                   evaluator_hashes=evaluator_hashes(), plan_sha256=sha(PLAN),
                   method='A3, four depths, centroid margin, patch .35, DI, equal L1',
                   target_feedback_for_selection=False, batch_size=2,
                   design='hypothesis-driven exploratory fixed-cohort comparison; not untouched confirmation')
    # A documented reporting-only repair must preserve the scientific lock hash
    # and every other source hash. Only an explicitly recorded exact SHA pair is accepted.
    repair = ROOT / 'audits/reporting_source_repair.json'
    if repair.exists():
        change = json.loads(repair.read_text())
        key = 'proxy_study_protocol.py'
        if payload['source_hashes'][key] != change['new_sha256']:
            raise RuntimeError('Unrecognized protocol source after reporting repair')
        payload['source_hashes'][key] = change['original_sha256']
    freeze(ROOT / 'method_lock.json', payload)
    return payload


def report(rows, state):
    table = []
    directed = []
    for r in rows:
        item = r['item']; c = item['config']
        try:
            e = checked(item)
            complete = e['evaluation_complete']
            status = 'COMPLETE' if complete else 'ATTACK_COMPLETE' if e['attack_complete'] else 'PARTIAL' if e['attack_images'] else 'QUEUED'
            table.append(dict(template=r['template'], tier=r['tier'], target=c['target'],
                              proxies=' + '.join(c['models']), K=len(c['models']), status=status,
                              attack_images=e['attack_images'], evaluated_images=e['images'],
                              hits=sum(x['hits'] for x in e['rows']) if complete else None,
                              denominator=150 if complete else None,
                              ASR=sum(x['hits'] for x in e['rows']) / 150 if complete else None,
                              source=item['source'], condition=item['measurement_condition'],
                              attack_seconds=e['attack_seconds'], peak_reserved_gib=e['peak_reserved_gib'],
                              mean_gradient_cosine=e['mean_gradient_cosine'],
                              negative_gradient_fraction=e['negative_cosine_fraction'],
                              cancellation=e['mean_cancellation'],
                              error=state.get('blocked', {}).get(c['id'], '')))
            directed.extend({**x, 'template': r['template'], 'target': c['target']} for x in e['rows'])
        except Exception as exc:
            table.append(dict(template=r['template'], tier=r['tier'], target=c['target'],
                              proxies=' + '.join(c['models']), K=len(c['models']), status='BLOCKED', error=str(exc)))
    from proxy_selection_tables import writecsv
    writecsv(ROOT / 'reports/completion.csv', table)
    writecsv(ROOT / 'reports/directed_results.csv', directed)
    lines = ['# 18-hour proxy composition study', '',
             'Exploratory fixed-cohort comparisons; no target-feedback tuning. Missing results are N/A.',
             'Hypotheses: heterogeneity versus same-family redundancy; identity versus proxy count; marginal benefit versus compute cost.',
             '', f"Complete: {sum(x['status']=='COMPLETE' for x in table)}/{len(table)}; elapsed hours: {( __import__('time').time()-state['started'])/3600:.2f}.", '',
             '| Template | Target | K | Status | Hits/N | ASR |', '|---|---|---:|---|---|---|']
    for x in table:
        score = f"{x['ASR']:.1%}" if x.get('ASR') is not None else 'N/A'
        lines.append(f"| {x['template']} | {x['target']} | {x['K']} | {x['status']} | {x.get('hits')}/{x.get('denominator')} | {score} |")
    lines += ['', 'Do not infer causal effects of training objectives from different checkpoints alone. Same count does not imply same compute. Final target outcomes never change this queue.',
              'Raw PNGs and checkpoints stay on the persistent root disk; only compact measurements/reports should be published.',
              'Full paired contrasts are generated when both members have complete, matching image cohorts.']
    (ROOT / 'reports/live_report.md').write_text('\n'.join(lines) + '\n')
    return table
