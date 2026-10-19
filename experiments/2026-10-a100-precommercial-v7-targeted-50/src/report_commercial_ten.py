"""Recount frozen CLI observations and inspect saved surrogate traces, CPU only."""
import csv
import json
import math
from pathlib import Path
import statistics

from commercial_cli_adapters import parse_label, json_events
from prepare_commercial_ten import BASE, DEST
from proxy_study_protocol import OUT, sha, atomic_json


def wilson(hits,n):
    if not n:return None
    z=1.959963984540054;p=hits/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-half),min(1,center+half)]


def verified_record(rid,request):
    folder=DEST/'codex/gpt-6.1-sol'/rid;p=folder/'result.json'
    if not p.exists():return None
    d=json.loads(p.read_text())
    expected=dict(image_sha256=request['image_sha256'],prompt_sha256=request['prompt_sha256'],
                  client='codex-cli 0.163.0',model='gpt-6.1-sol')
    if d['identity']!=expected or d['parsed_label']!=parse_label(d['raw_output']):
        raise RuntimeError('Cached observation identity/parser mismatch')
    if sha(BASE/'blinded'/rid/'image.png')!=request['image_sha256']:
        raise RuntimeError('Frozen input changed')
    events=json_events((folder/'stdout.jsonl').read_text())
    threads=[e['thread_id'] for e in events if e.get('type')=='thread.started']
    items=[e['item'] for e in events if e.get('type')=='item.completed']
    if len(threads)!=1 or not any(e.get('type')=='turn.completed' for e in events):
        raise RuntimeError('Missing fresh completed context')
    if any(e.get('type') not in ['agent_message','reasoning','error'] for e in items):
        raise RuntimeError('Unexpected tool in target trace')
    messages=[e.get('text','').strip() for e in items if e.get('type')=='agent_message']
    if not messages or messages[-1]!=d['raw_output'].strip():raise RuntimeError('Raw answer/trace mismatch')
    return dict(d,session_id=threads[0],reused=(folder/'reuse_provenance.json').exists())


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    manifest=json.loads((BASE/'private/manifest.json').read_text())
    requests={r['request_id']:r for r in manifest['requests']}
    rows=[];diagnostics=[];sessions=[];elapsed=[];new_elapsed=[]
    for c in manifest['cases']:
        pair={k:verified_record(rid,requests[rid]) for k,rid in c['requests'].items()}
        for d in pair.values():
            if d:
                sessions.append(d['session_id'])
                # One historical trace was recovered without a request start time.
                # Preserve the observation but do not invent its runtime.
                if 'started' in d:
                    elapsed.append(d['finished']-d['started'])
                    if not d['reused']:new_elapsed.append(d['finished']-d['started'])
        complete=all(pair.values());clean=pair['clean'];adv=pair['adversarial']
        reused=all(d and d['reused'] for d in pair.values())
        rows.append(dict(transition=c['transition'],image_id=c['image_id'],source_label=c['source_label'],
                 target_label=c['target_label'],complete=complete,previous_pilot_pair=reused,
                 clean_raw=clean['raw_output'] if clean else None,adv_raw=adv['raw_output'] if adv else None,
                 clean_label=clean['parsed_label'] if clean else None,adv_label=adv['parsed_label'] if adv else None,
                 clean_correct=clean['parsed_label']==c['source_label'] if clean else None,
                 target_hit=adv['parsed_label']==c['target_label'] if adv else None,
                 adv_source_retained=adv['parsed_label']==c['source_label'] if adv else None))
        statepath=OUT/c['source_state'];state=json.loads(statepath.read_text())
        if state['target_access']:raise RuntimeError('Target access in attack provenance')
        i=state['image_ids'].index(c['image_id']);history=state['trace'][0]
        allpair=[p[i] for step in history for p in step['pairwise_cosines']]
        for m,name in enumerate(state['models']):
            diagnostics.append(dict(transition=c['transition'],image_id=c['image_id'],proxy=name,
                       first_loss=history[0]['losses'][m][i],step50_loss=history[-1]['losses'][m][i],
                       first_margin=history[0]['margins'][m][i],step50_margin=history[-1]['margins'][m][i],
                       step50_margin_positive=history[-1]['margins'][m][i]>0,
                       negative_pair_fraction=sum(v<0 for v in allpair)/len(allpair) if allpair else None,
                       mean_pair_cosine=statistics.mean(allpair) if allpair else None,
                       mean_cancellation_ratio=statistics.mean(step['cancellation_ratio'][i] for step in history),
                       source_state=str(statepath),source_state_sha256=sha(statepath),
                       scope='DI/transformed pre-update iteration traces; not final-PNG margins or target logits'))
    if len(sessions)!=len(set(sessions)):raise RuntimeError('Target session reused')
    complete=all(r['complete'] for r in rows)
    hits=sum(r['target_hit'] is True for r in rows)
    new=[r for r in rows if not r['previous_pilot_pair']]
    cleanvalid=[r for r in rows if r['clean_correct'] is True]
    summary=dict(status='COMPLETE' if complete else 'PARTIAL',model='gpt-6.1-sol',reasoning_effort='low',
        hits=hits if complete else None,denominator=10,ASR=hits/10 if complete else None,
        goal_hits=6,goal_met=hits>=6 if complete else None,
        clean_correct=sum(r['clean_correct'] is True for r in rows),
        clean_correct_conditional_hits=sum(r['target_hit'] is True for r in cleanvalid) if complete else None,
        clean_correct_conditional_denominator=len(cleanvalid) if complete else None,
        completed_requests=len(sessions),planned_requests=20,
        new_requests=sum(not (DEST/'codex/gpt-6.1-sol'/rid/'reuse_provenance.json').exists() for rid in requests if (DEST/'codex/gpt-6.1-sol'/rid/'result.json').exists()),
        reused_requests=sum((DEST/'codex/gpt-6.1-sol'/rid/'reuse_provenance.json').exists() for rid in requests),
        previously_unobserved_image_hits=sum(r['target_hit'] is True for r in new) if complete else None,
        previously_unobserved_image_denominator=7,
        new_request_wall_seconds=sum(new_elapsed),known_request_wall_seconds=sum(elapsed),runtime_records_available=len(elapsed),
        wilson95=wilson(hits,10) if complete else None,
        target_feedback_for_tuning=False,small_development_sample=True,
        manifest_sha256=sha(BASE/'private/manifest.json'))
    report=DEST/'reports';report.mkdir(parents=True,exist_ok=True)
    atomic_json(report/'summary.json',summary)
    write_csv(report/'image_results.csv',rows);write_csv(report/'proxy_diagnostics.csv',diagnostics)
    directions=[]
    for tid in ['T01','T03','T05','T07','T09']:
        group=[r for r in rows if r['transition']==tid];done=all(r['complete'] for r in group)
        directions.append(dict(transition=tid,hits=sum(r['target_hit'] is True for r in group) if done else None,
                               denominator=2,ASR=sum(r['target_hit'] is True for r in group)/2 if done else None))
    write_csv(report/'direction_results.csv',directions)
    lines=['# Ten-image GPT transfer check','',f"Status: {summary['status']}; gpt-6.1-sol via Codex CLI0.163.0, effort low.",
        f"Targeted hits: {summary['hits']}/10. ASR: {summary['ASR']}. Goal >=6/10 met: {summary['goal_met']}.",
        f"Clean correct: {summary['clean_correct']}/10. New-image hits: {summary['previously_unobserved_image_hits']}/7; three images were observed in the earlier pilot.",
        f"Requests: {len(sessions)}/20 verified, {summary['new_requests']} new and {summary['reused_requests']} reused. No negative-result retries.",
        f"Sum of new request durations: {summary['new_request_wall_seconds']:.1f}s. Wilson95 interval over10: {summary['wilson95']} (descriptive small-sample interval, not population validation).",'',
        '| Direction | Targeted hits / fixed denominator |','|---|---|']
    lines += [f"| {r['transition']} | {r['hits']}/2 |" for r in directions]
    lines += ['', '## Saved surrogate optimization diagnostics', '',
        'Each proxy row summarizes10 images. Margin is target cosine minus strongest competing class, averaged across four layers. These are step50 transformed/pre-update traces, not final frozen-PNG inference. Loss and margin success are different quantities.', '',
        '| Proxy | Positive step50 margin /10 | Mean first margin | Mean step50 margin | Loss decreased /10 |','|---|---|---|---|---|']
    for name in dict.fromkeys(d['proxy'] for d in diagnostics):
        group=[d for d in diagnostics if d['proxy']==name]
        lines.append(f"| {name} | {sum(d['step50_margin_positive'] for d in group)}/10 | {statistics.mean(d['first_margin'] for d in group):.5f} | {statistics.mean(d['step50_margin'] for d in group):.5f} | {sum(d['step50_loss']<d['first_loss'] for d in group)}/10 |")
    unique=diagnostics[::3]
    lines += ['', f"Mean negative gradient-pair fraction across50 iterations: {statistics.mean(d['negative_pair_fraction'] for d in unique):.4f}; mean cancellation ratio: {statistics.mean(d['mean_cancellation_ratio'] for d in unique):.4f}.", '',
        '## Interpretation and next controlled tests', '',
        'The recorded proxy losses/margins diagnose optimization progress; they do not prove that a closed target must change its answer. Negative proxy margins would support incomplete surrogate optimization. Positive feature margins with retained GPT source answers support a transfer gap in this sample, without identifying a unique cause.',
        'Proxy gradient conflicts, native preprocessing differences and feature-to-language decision mismatch are plausible mechanisms. Their causal contributions are not established by these saved traces. Closed-target gradients, weights and decision margins are unavailable.',
        'Any next method must be chosen on separate held-out surrogate validation, with fixed compute and controlled loss/layer/fusion changes. Do not select proxy combinations, images, seeds or prompts using these GPT responses. The existing three-image negatives remain in the ten-image total.',
        'This is an exploratory development-cohort check with n=2/direction. It is not the final150-image benchmark, open-ended VQA, or evidence that a 60% population transfer rate has been achieved.']
    (report/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
