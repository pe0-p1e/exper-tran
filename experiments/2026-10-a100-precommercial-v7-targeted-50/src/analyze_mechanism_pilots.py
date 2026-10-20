"""CPU-only checks and trace measurements; never reloads any model."""
import argparse
import json
import time
import numpy as np
from PIL import Image
from pp4_protocol import OUT,EXP,sha,atomic_json,development_ids
from pp4_campaign import table

ROOT=OUT/'pull_push_mechanism_pilots'


def analyze():
    lock=json.loads((ROOT/'protocol_lock.json').read_text());rows=[];sources=[];validated=0;pixels=0
    expected={(tid,iid) for tid,ids in development_ids().items() for iid in ids}
    for item in lock['plan']['ladder']:
        c=item['condition'];seen=set();states=[]
        for path in sorted((OUT/'attacks/dev'/c['name']).glob('T*/batch_*/state.json')):
            d=json.loads(path.read_text());states.append(d);sources.append(dict(path=str(path),sha256=sha(path)))
            if d['steps']!=50 or d['seed']!=42 or d['target_access'] or d['models']!=c['models']:
                raise RuntimeError('Invalid attack provenance')
            if d['config']!={**c['options'],'fusion':'equal','input_diversity':True,'smoothing':False}:
                raise RuntimeError('Wrong scientific options')
            for iid,clean,adv in zip(d['image_ids'],d['clean_paths'],d['adv_paths']):
                key=(d['transition_id'],iid)
                if key not in expected or key in seen:raise RuntimeError('Cohort mismatch or duplicate')
                seen.add(key)
                a=np.asarray(Image.open(OUT/clean).convert('RGB')).astype(np.int16)
                b=np.asarray(Image.open(OUT/adv).convert('RGB')).astype(np.int16)
                if a.shape!=b.shape or a.shape!=(224,224,3) or np.max(np.abs(a-b))>16:
                    raise RuntimeError('Pixel-bound violation')
                pixels+=1
            for rel,h in d['png_hashes'].items():
                if sha(OUT/rel)!=h:raise RuntimeError('PNG hash mismatch')
        # Independently recompute validation numerators rather than trust reports.
        native=[]
        for path in sorted((ROOT/'validation'/c['id']).glob('*.json')):
            v=json.loads(path.read_text());obs=v['images']
            if len(obs)!=10 or {(r['transition_id'],r['image_id']) for r in obs}!=expected:
                raise RuntimeError('Validation cohort differs')
            if v['model'] in c['models']:raise RuntimeError('Validator is an attack checkpoint')
            hits=sum(r['adv_label']==r['target_label'] for r in obs)
            if hits!=v['hits'] or v['denominator']!=10 or any(r['target_hit']!=(r['adv_label']==r['target_label']) for r in obs):
                raise RuntimeError('Incorrect metric numerator/denominator')
            validated+=1;native+=obs
        traces=[t for d in states for restart in d['trace'] for t in restart]
        for proxy_index,model in enumerate(c['models']):
            first=[float(np.mean(restart[0]['losses'][proxy_index])) for d in states for restart in d['trace']]
            last=[float(np.mean(restart[-1]['losses'][proxy_index])) for d in states for restart in d['trace']]
            margin_first=[float(np.mean(restart[0]['margins'][proxy_index])) for d in states for restart in d['trace']]
            margin_last=[float(np.mean(restart[-1]['margins'][proxy_index])) for d in states for restart in d['trace']]
            norms=[x for t in traces for x in t['gradient_l1'][proxy_index]]
            rows.append(dict(group=item['group'],label=item['label'],proxy=model,attack_images=len(seen),
                initial_stochastic_view_loss=float(np.mean(first)) if first else None,
                last_stochastic_view_loss=float(np.mean(last)) if last else None,
                initial_margin=float(np.mean(margin_first)) if margin_first else None,
                last_margin=float(np.mean(margin_last)) if margin_last else None,
                mean_raw_gradient_L1=float(np.mean(norms)) if norms else None,
                caveat='Different DI views at endpoints; layer-specific loss scales differ; not a fixed-view optimization gap'))
    table(ROOT/'reports/proxy_optimization_traces.csv',rows)
    fusion=[]
    for path in sorted((ROOT/'fusion_diagnostics').glob('*.json')):
        d=json.loads(path.read_text());calls=d['calls']
        c=next(r['condition'] for r in lock['plan']['ladder'] if r['condition']['id']==d['condition'])
        for i,model in enumerate(c['models']):
            fractions=[v for call in calls for v in call['raw_contribution_fraction'][i]]
            cosine=[v for call in calls for v in call['raw_vs_normalized_cosine']]
            fusion.append(dict(condition=c['id'],proxy=model,calls=len(calls),
                mean_raw_L1_share=float(np.mean(fractions)),mean_direction_cosine=float(np.mean(cosine)),
                process_record=str(path),caveat='Raw norm fraction is not signed causal contribution; resumed processes are separate records'))
    table(ROOT/'reports/raw_gradient_dominance.csv',fusion)
    audits=json.loads((ROOT/'audits/model_dtype_audit.json').read_text()) if (ROOT/'audits/model_dtype_audit.json').exists() else []
    if any(a['status']!='pass' for a in audits):raise RuntimeError('BF16 audit failure')
    atomic_json(ROOT/'reports/measurement_validation.json',dict(at=time.time(),status='PASS_WITH_EXPLORATORY_CAVEATS',
        checked_attack_state_files=len(sources),checked_clean_adv_image_pairs=pixels,checked_validation_files=validated,
        BF16_load_events=len(audits),sources=sources,full_campaign_complete=(ROOT/'completion.json').exists(),
        model_inference_rerun=False,independent_images_per_condition=10,validators_per_condition=3,
        caveats=['Previously inspected development cohort','Checkpoint-held-out, not whole-family-held-out validators',
            'No equivalence inference from small paired differences','No new target DeltaR, CKA or RSA inference in this CPU-only check']))
    # A compact decision readout, preserving missing cells rather than imputing zeros.
    import csv
    def read(path):
        return list(csv.DictReader(path.open())) if path.exists() else []
    conditions=read(ROOT/'reports/conditions.csv')
    di=read(OUT/'pull_push_di_removal/reports/paired_DI.csv')
    ratios=read(OUT/'pull_push_composition_ablation/reports/paired_weights.csv')
    lines=['# Initial mechanism findings','',
        'Exploratory only:10 previously inspected development images per condition, evaluated by3 held-out checkpoints. These are30 correlated model-image observations, not30 independent images. No final-target or commercial output selected these arms.','',
        '| Composition | Intervention | Status | Hits /30 observations |','|---|---|---|---|']
    for r in conditions:
        lines.append(f"| {r['group']} | {r['label']} | {r['status']} | {r['hits'] or 'N/A'}/30 |")
    lines+=['','## Existing DI removal results','']
    for r in di:
        lines.append(f"Composition {r['group']}: DI-on {r['DI_on_hits']}/30; off {r['DI_off_hits']}/30. Paired removal delta {float(r['off_minus_on']):.1%}; exploratory image-cluster95% interval [{float(r['image_cluster_CI_low']):.1%},{float(r['image_cluster_CI_high']):.1%}].")
    lines+=['','## Balance and proxy evidence already available','',
        f"Reuse the completed13-composition ratio study ({len(ratios)} matched ratio comparisons available), and the original five-ratio bounded diagnostic. Do not rerun either automatically. Balanced coefficients are interpretable but do not imply balanced pixel gradients.",
        'For proxy complementarity use matched member singles from that same composition study. Low CKA is a candidate predictor, not evidence of a causal benefit or pixel-gradient orthogonality. No new CKA/RSA or fixed-view target representation extraction was performed in this CPU pass.',
        '', '## What should determine the next larger experiment?', '',
        '* DI: preserve the completed signal; no further DI search now.',
        '* Gradient normalization: examine the paired score together with raw norm dominance/direction measurements. A small score difference does not prove equivalence or a necessary mechanism.',
        '* Layers: determine whether75% alone,50/75% or50/75/100% retains the four-layer effect. These arms do not identify a globally best subset; follow up with a small equal-count identity comparison if warranted.',
        '* Confirm promising contrasts on an independently frozen cohort and predeclared seeds before expanding original full tables. Do not choose a configuration using final-target feedback.',
        '* Compare the same images and report direction-specific outcomes and image-cluster uncertainty; any later many-contrast inferential analysis needs multiplicity handling.',
        '', '## Validation scope', '',
        f"CPU checks cover {len(sources)} attack batches, {pixels} clean/adv image pairs, {validated} native validation files and {len(audits)} strict-BF16 load records available at this snapshot. Bytewise pixel bounds, PNG hashes, cohort uniqueness,50-step settings, no target access and native hit numerators were checked. Missing rows remain missing.",
        'Optimization CSV endpoints use different DI views and different layer-specific objectives; do not interpret their raw loss differences as comparable fixed-view progress.',
        'Raw L1 share is a magnitude diagnostic, not signed causal attribution. Source condition identities and per-image observations remain in the companion reports.']
    (ROOT/'reports/initial_findings.md').write_text('\n'.join(lines)+'\n')
    print('PASS CPU provenance, pixel and metric checks:',len(sources),'batches;',pixels,'image pairs;',validated,'validation files')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    if args.watch:
        while not (ROOT/'completion.json').exists():time.sleep(15)
    analyze()
