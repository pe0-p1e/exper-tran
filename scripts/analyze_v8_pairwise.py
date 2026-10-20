#!/usr/bin/env python3
"""CPU-only paired model/layer diagnostics. Never schedules inference or selects methods."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

DEFAULT_ROOT = Path('/home/ubuntu/exper-tran/outputs/a100_precommercial_v8_joint_selection24h')


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def atoms(c):
    return tuple(sorted((model, layer) for model, layers in c['layers'].items() for layer in layers))


def context(c):
    # Model order follows the sorted atomic pool; layers/order are the interventions.
    return json.dumps({k: v for k, v in c.items() if k not in ('id', 'label', 'layers', 'model_order')}, sort_keys=True)


def checked_rows(e, manifest):
    expected = {r['image_id']: r for r in manifest['images']}
    rr = e.get('images', [])
    if not e.get('complete') or e.get('N') != len(expected) or len(rr) != len(expected):
        raise ValueError('Incomplete evaluation; missing results are not zero')
    actual = {r['image_id']: r for r in rr}
    if len(actual) != len(rr) or actual.keys() != expected.keys():
        raise ValueError('Duplicate or mismatched image IDs')
    for iid, r in actual.items():
        if any(r[k] != expected[iid][k] for k in ('source', 'target', 'transition')):
            raise ValueError('Cohort labels/directions differ')
        if type(r['target_hit']) is not bool or r['target_hit'] != (r['adv_label'] == r['target']):
            raise ValueError('Target-hit scoring mismatch')
    return actual


def comparison(joint, left, right):
    """Aligned binary outcomes; clean response must be identical for all three methods."""
    if not joint or joint.keys() != left.keys() or joint.keys() != right.keys():
        raise ValueError('Unpaired or empty outcomes')
    rows = []
    lh, rh = sum(r['target_hit'] for r in left.values()), sum(r['target_hit'] for r in right.values())
    best = left if lh >= rh else right
    for iid in sorted(joint):
        j, u, v = joint[iid], left[iid], right[iid]
        for key in ('target', 'source', 'transition', 'clean_label'):
            if not j[key] == u[key] == v[key]:
                raise ValueError('Mismatched paired cohort or cached clean predictions')
        hj, hu, hv = int(j['target_hit']), int(u['target_hit']), int(v['target_hit'])
        hc = int(j['clean_label'] == j['target'])
        rows.append(dict(image_id=iid, transition=j['transition'], joint_hit=hj, left_hit=hu, right_hit=hv,
                         clean_target_hit=hc, interaction=hj-hu-hv+hc,
                         joint_only_hit=int(hj and not hu and not hv),
                         gain_vs_best=int(hj and not best[iid]['target_hit']),
                         loss_vs_best=int(not hj and best[iid]['target_hit'])))
    return rows


def summarize(rows):
    n = len(rows)
    hits = {k: sum(r[k] for r in rows) for k in ('joint_hit', 'left_hit', 'right_hit', 'clean_target_hit')}
    return dict(N=n, joint_hits=hits['joint_hit'], left_hits=hits['left_hit'], right_hits=hits['right_hit'],
                clean_target_hits=hits['clean_target_hit'], joint_ASR=hits['joint_hit']/n,
                left_ASR=hits['left_hit']/n, right_ASR=hits['right_hit']/n,
                clean_target_ASR=hits['clean_target_hit']/n, I=sum(r['interaction'] for r in rows)/n,
                joint_only_hits=sum(r['joint_only_hit'] for r in rows),
                gains_vs_best=sum(r['gain_vs_best'] for r in rows), losses_vs_best=sum(r['loss_vs_best'] for r in rows))


def bootstrap(rows, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    # Paired image resampling within fixed directions; never pool validator observations as independent.
    groups = [[r['interaction'] for r in rows if r['transition'] == t] for t in sorted({r['transition'] for r in rows})]
    values = sum(rng.choice(g, (4000, len(g)), replace=True).sum(axis=1) for g in groups)/len(rows)
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def heatmaps(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for family in sorted({r['family'] for r in records}):
        rr = [r for r in records if r['family'] == family and r['split'] == 'screen' and r['label'] == 'joint_pair']
        if not rr:
            continue
        units = sorted({r[k] for r in rr for k in ('left_unit', 'right_unit')})
        matrix = np.full((len(units), len(units)), np.nan)
        for r in rr:
            i, j = units.index(r['left_unit']), units.index(r['right_unit'])
            matrix[i, j] = matrix[j, i] = r['I']
        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.colormaps['RdBu_r'].copy(); cmap.set_bad('#dddddd')
        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(units)), units, rotation=70, ha='right', fontsize=8)
        ax.set_yticks(range(len(units)), units, fontsize=8)
        for i in range(len(units)):
            for j in range(len(units)):
                ax.text(j, i, 'N/A' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:+.2f}', ha='center', va='center', fontsize=8)
        ax.set_title(f'{family}: paired interaction I; gray = unmeasured/diagonal\nDevelopment outcomes; no compute-matched causality claim')
        fig.colorbar(im, ax=ax, label='T(pair) - T(left) - T(right) + T(clean)')
        fig.tight_layout(); fig.savefig(out/f'pairwise_{family}.png', dpi=150); plt.close(fig)


def generate(root):
    out = root/'reports'
    cs = [read(p) for p in sorted((root/'conditions').glob('*.json'))]
    singles = {(context(c), atoms(c)): c for c in cs if len(atoms(c)) == 1}
    complete, individual, directions, pending = [], [], [], []
    for c in cs:
        units = atoms(c)
        if len(units) != 2 or c['steps'] != 50:
            continue
        try:
            u, v = [singles[(context(c), (a,))] for a in units]
            manifest = read(root/'manifests'/f"{c['split']}.json")
            evaluations = []
            for x in (c, u, v):
                ev = read(root/'validation'/f"{x['id']}.json")
                science = {k: val for k, val in x.items() if k not in ('id', 'label')}
                expected_hash = hashlib.sha256(json.dumps(science, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if ev.get('science') != expected_hash or ev.get('model') != c['validator']:
                    raise ValueError('Evaluation scientific identity mismatch')
                evaluations.append(checked_rows(ev, manifest))
            rows = comparison(*evaluations)
            lo, hi = bootstrap(rows, int(hashlib.sha256(c['id'].encode()).hexdigest()[:8], 16))
            summary = dict(condition=c['id'], label=c['label'], family=c['family'], split=c['split'],
                           left_unit=f'{units[0][0]}:block{units[0][1]}', right_unit=f'{units[1][0]}:block{units[1][1]}',
                           left_condition=u['id'], right_condition=v['id'], pair_type='within_model' if len(c['layers']) == 1 else 'cross_model',
                           **summarize(rows), I_bootstrap_low=lo, I_bootstrap_high=hi)
            cost_csv = out/'conditions.csv'
            costs = list(csv.DictReader(cost_csv.open())) if cost_csv.exists() else []
            for name, cc in [('joint', c), ('left', u), ('right', v)]:
                cost = next((r for r in costs if r['id'] == cc['id']), {})
                for key in ('backwards', 'attack_seconds', 'peak_reserved_gib'):
                    summary[f'{name}_{key}'] = cost.get(key, '')
            complete.append(summary)
            individual.extend(dict(condition=c['id'], family=c['family'], **r) for r in rows)
            for t in sorted({r['transition'] for r in rows}):
                directions.append(dict(condition=c['id'], family=c['family'], transition=t, **summarize([r for r in rows if r['transition'] == t])))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            pending.append(dict(condition=c['id'], status='PENDING_OR_INVALID', reason=str(exc)))
    if complete:
        table(out/'pairwise_complementarity.csv', complete, list(complete[0]))
        table(out/'pairwise_per_image.csv', individual, list(individual[0]))
        table(out/'pairwise_per_direction.csv', directions, list(directions[0]))
        heatmaps(out, complete)
    write_json(out/'pairwise_analysis_status.json', dict(complete_pairs=len(complete), pending=pending,
               expected_exhaustive_pairs_per_validator=15, formula='I=T(pair)-T(left)-T(right)+T(clean)',
               uncertainty='4000 paired image bootstrap resamples stratified by direction; exploratory, post-selection, small N; not equivalence/causal proof',
               selection_feedback=False))
    (out/'PAIRWISE_INTERPRETATION.md').write_text('''# Model–layer pairwise control

Six development-selected atomic units per validator produce 15 real joint attacks: three within-model layer pairs and twelve cross-model pairs. No union of singleton attack images is treated as an ensemble.

I = T(pair) − T(left) − T(right) + T(clean). All four scores use the same frozen image IDs, labels, validator, prompt and unconditional denominator. Cached clean responses must agree. This additive interaction on a bounded binary metric can be negative even when the pair beats both members (ceiling effects); it is not a standalone selection score or proof of semantic complementarity. Also inspect joint-only hits and gains/losses against the stronger singleton.

Uncertainty resamples paired images within directions. These are small, development-selected comparisons, not untouched confirmatory significance tests. Missing/partial results are not zeros. Within-model two-layer pairs and cross-model pairs differ in backward cost; report measured runtime, calls and peak memory before attributing gains to complementarity. Timing may depend on concurrent workloads. No new loss, inference or commercial feedback is introduced by this analysis.
''')
    return dict(complete_pairs=len(complete), pending_pairs=len(pending))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    print(json.dumps(generate(parser.parse_args().root)))
