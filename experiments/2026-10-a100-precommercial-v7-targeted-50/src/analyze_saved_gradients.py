"""CPU-only compatibility from saved common-view gradients; no new inference."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from proxy_study_protocol import OUT,ROOT,plan,checked,atomic_json
from finalize_v7 import writecsv


def main():
    target=ROOT/'mechanism_gradients';target.mkdir(parents=True,exist_ok=True)
    reference=list(csv.DictReader((ROOT/'mechanism_reference/cross_model_cka_rsa.csv').open()))
    ref={frozenset([r['left_model'],r['right_model']]):r for r in reference}
    summary=[];pairs=[];proxies=[];seen=set()
    for row in plan():
        item=row['item'];condition=item['measurement_condition'];victim=item['config']['target']
        if (condition,victim) in seen:continue
        seen.add((condition,victim));states=[]
        for p in sorted((OUT/'attacks/test'/condition).glob('T*/batch_*/state.json')):
            d=json.loads(p.read_text())
            if d.get('status')=='attack_complete' and d.get('trace'):states.append(d)
        if not states:continue
        names=states[0]['models'];count=sum(len(d['image_ids']) for d in states)
        all_cancel=[];first_cancel=[];trajectory=[]
        for pair_index,(i,j) in enumerate(itertools.combinations(range(len(names)),2)):
            first=[];last=[];all_cos=[]
            for d in states:
                # Report only restart1 histories for these fixed single-restart conditions.
                history=d['trace'][0]
                first.extend(history[0]['pairwise_cosines'][pair_index]);last.extend(history[-1]['pairwise_cosines'][pair_index])
                for step in history:all_cos.extend(step['pairwise_cosines'][pair_index])
            r=ref.get(frozenset([names[i],names[j]]),{})
            pairs.append(dict(condition=condition,target=victim,left_proxy=names[i],right_proxy=names[j],N=count,
                 first_step_mean_cosine=float(np.mean(first)),last_step_mean_cosine=float(np.mean(last)),
                 trajectory_mean_cosine=float(np.mean(all_cos)),negative_pair_fraction=float(np.mean(np.array(all_cos)<0)),
                 reference_CKA=r.get('reference_CKA'),reference_RSA=r.get('reference_RSA'),
                 reference_within_class_residual_CKA=r.get('within_class_residual_CKA')))
            trajectory.extend(all_cos)
        for d in states:
            for step in d['trace'][0]:all_cancel.extend(step['cancellation_ratio'])
            first_cancel.extend(d['trace'][0][0]['cancellation_ratio'])
        for i,name in enumerate(names):
            first_loss=[];last_loss=[];first_margin=[];last_margin=[]
            for d in states:
                a,b=d['trace'][0][0],d['trace'][0][-1]
                first_loss.extend(a['losses'][i]);last_loss.extend(b['losses'][i]);first_margin.extend(a['margins'][i]);last_margin.extend(b['margins'][i])
            proxies.append(dict(condition=condition,target=victim,proxy=name,N=count,
                  mean_first_loss=float(np.mean(first_loss)),mean_last_loss=float(np.mean(last_loss)),
                  loss_decreased_count=int((np.array(last_loss)<first_loss).sum()),
                  mean_first_margin=float(np.mean(first_margin)),mean_last_margin=float(np.mean(last_margin)),
                  positive_last_margin_count=int((np.array(last_margin)>0).sum())))
        e=checked(item)
        summary.append(dict(condition=condition,target=victim,K=len(names),available_attack_images=count,
              attack_complete=e['attack_complete'],evaluation_complete=e['evaluation_complete'],
              target_ASR=e.get('ASR') if e['evaluation_complete'] else None,
              first_step_cancellation=float(np.mean(first_cancel)),trajectory_cancellation=float(np.mean(all_cancel)),
              trajectory_gradient_negative_pair_fraction=float(np.mean(np.array(trajectory)<0)) if trajectory else None))
    writecsv(target/'condition_summary.csv',summary);writecsv(target/'pairwise_compatibility.csv',pairs);writecsv(target/'per_proxy_optimization.csv',proxies)
    atomic_json(target/'provenance.json',dict(conditions_with_saved_traces=len(summary),model_inference_calls=0,
           complete_and_partial_separated=True,first_step='Same adversarial initialization and shared DI view within each ensemble; not the unperturbed clean image',
           trajectory='Later trajectories differ across ensembles; do not treat later gradients as measured at the same input',
           margins='Layer-averaged transformed pre-update representation margins, not final PNG or output predictions',
           inference='Reference similarity and gradient alignment associations are descriptive; no causal claim or target-guided selection'))
    print('Saved',len(summary),'condition summaries,',len(pairs),'proxy pairs,',len(proxies),'optimization records.',flush=True)


if __name__=='__main__':main()
