#!/usr/bin/env python3
"""Independent receipt-level closeout checks and paired statistics; no inference."""
import argparse,csv,fcntl,itertools,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from paper_closeout.plan import OUT

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.tmp').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    path.with_suffix('.tmp').replace(path)

def wilson(h,n):
    if not n:return None,None
    z=1.959963984540054;p=h/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)

def paired(a,b,seed=42):
    if set(a)!=set(b):raise ValueError('Paired cohort identity mismatch')
    keys=sorted(a);x=np.array([a[k]['hit'] for k in keys],dtype=int);y=np.array([b[k]['hit'] for k in keys],dtype=int)
    gain=int(((x==1)&(y==0)).sum());loss=int(((x==0)&(y==1)).sum())
    delta=x-y;rng=np.random.default_rng(seed);boot=np.zeros(10000)
    # Stratified image resampling preserves the five predeclared direction counts.
    for tid in C.TIDS:
        indices=np.array([i for i,k in enumerate(keys) if a[k]['transition']==tid])
        if len(indices):boot+=delta[rng.choice(indices,size=(len(boot),len(indices)),replace=True)].sum(axis=1)
    ci=np.quantile(boot/len(keys),[.025,.975])
    return dict(N=len(keys),a_hits=int(x.sum()),b_hits=int(y.sum()),a_only=gain,b_only=loss,
        difference_pp=float(delta.mean()*100),CI95_low_pp=float(ci[0]*100),CI95_high_pp=float(ci[1]*100),
        McNemar_exact_p=float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.,
        bootstrap='10000 paired image resamples, stratified by direction; seed42',test_unit='image, not model-image pooled observations')

def run():
    C.check_lock()
    report=OUT/'reports';errors=[];completion=[];cohorts={};groups=defaultdict(dict);directions=[];receipts=[]
    final=C.read(OUT/'final_method_lock.json');ratio=C.read(OUT/'pull_push_ablation_lock.json')
    # Read canonical final/ratio target evaluator files independently of analysis CSVs.
    specs=[]
    for fam,methods in final['families'].items():
        for method,v in methods.items():specs.append(('final150',fam,method,v['condition']))
    for fam,conditions in ratio.items():
        for c in conditions:specs.append(('ratio30',fam,'pull_'+str(c['pull']),c))
    for fam,methods in C.read(OUT/'followup_conditions.json',{}).items():
        for method,splits in methods.items():
            for split,c in splits.items():specs.append(('development_'+split,fam,method,c))
    seen_receipts=set()
    for study,fam,method,c in specs:
        C.legal(c['layers'],C.VALIDATORS[fam]);mode='evaluate_large_'+fam;target=C.LARGE_EVAL_MODES[mode]
        manifest=C.cohort(c['split'])['images'];expected={r['image_id']:r for r in manifest}
        if study.startswith('development_'):
            target=C.VALIDATORS[fam];ev=C.read(C.eval_path(c))
        else:ev=C.read(C.ROOT/'large_validation'/target/(c['id']+'.json'))
        if not ev:
            # Historical directory naming is model name (verify rather than infer).
            candidates=list((C.ROOT/'large_validation').glob('*/'+c['id']+'.json'))
            matches=[C.read(p) for p in candidates if C.read(p).get('model')==target]
            ev=matches[0] if len(matches)==1 else None
        items=(ev or {}).get('images',[]);data={}
        for x in items:
            iid=x['image_id'];r=expected.get(iid)
            if iid in data:errors.append('Duplicate evaluation: '+c['id']+'/'+iid)
            if not r:errors.append('Unexpected evaluation image: '+iid);continue
            hit=x.get('adv_label')==r['target'];clean=x.get('clean_label')==r['source']
            if hit!=x.get('target_hit'):errors.append('Hit/parser mismatch: '+c['id']+'/'+iid)
            data[iid]=dict(hit=hit,clean_correct=clean,clean_target=x.get('clean_label')==r['target'],clean_valid=x.get('clean_label') is not None,adv_valid=x.get('adv_label') is not None,transition=r['transition'])
            p=C.image_receipt(c,iid)
            if str(p) not in seen_receipts:
                seen_receipts.add(str(p));valid=C.valid_receipt(p,c);s=C.read(p,{})
                if not valid:errors.append('Invalid PNG/provenance: '+str(p))
                from PIL import Image
                clean=np.array(Image.open(s['clean']).convert('RGB'),dtype=np.int16);adv=np.array(Image.open(s['adv']).convert('RGB'),dtype=np.int16)
                canonical=np.array(Image.open(r['path']).convert('RGB'),dtype=np.int16)
                if not np.array_equal(clean,canonical):errors.append('Receipt clean differs from frozen cohort PNG: '+str(p))
                pixel_linf=int(np.abs(adv-clean).max())
                if clean.shape!=(224,224,3) or adv.shape!=clean.shape or pixel_linf>16:errors.append('Saved PNG pixel bound/shape failed: '+str(p))
                if s.get('steps')!=50 or s.get('seed')!=42:errors.append('Attack steps/seed mismatch: '+str(p))
                receipts.append(dict(condition=c['id'],image_id=iid,valid=valid,linf=s.get('linf_png'),linf_uint8=pixel_linf,target_access=s.get('target_access'),seconds=s.get('seconds'),backwards=s.get('model_backward_calls'),peak_reserved_gib=s.get('peak_reserved_gib')))
        complete=len(data)==len(expected) and set(data)==set(expected) and bool((ev or {}).get('complete'))
        if ev and (ev.get('N')!=len(expected) or ev.get('hits')!=sum(x['hit'] for x in data.values())):errors.append('Summary denominator/numerator mismatch: '+c['id'])
        key=(study,target);groups[key][method]=data;cohorts[key]=set(expected)
        completion.append(dict(study=study,model=target,method=method,observed=len(data),N=len(expected),status='COMPLETE' if complete else 'PARTIAL',hits=sum(x['hit'] for x in data.values())))
    for study in ['commercial50','commercial150','ratio30']:
        for pack in sorted((OUT/study/'packs').glob('*/*')):
            manifest=C.read(pack/'private/manifest.json',{});model=pack.parent.name;data={}
            for r in manifest.get('cases',[]):
                records={k:C.read(OUT/study/'responses'/model/rid/'result.json') for k,rid in r['requests'].items()}
                if not all(records.values()):continue
                for kind,response in records.items():
                    rid=r['requests'][kind]
                    if response.get('request_id')!=rid or response['identity']['model']!=model or response.get('tool_calls')!=0:errors.append('Commercial identity/tools mismatch: '+rid)
                    raw=response.get('raw_output','');line=next((line.strip() for line in raw.splitlines() if line.strip()),'')
                    parsed=int(line)+1 if line in tuple('0123456789') else None
                    if parsed!=response.get('parsed_label'):errors.append('Raw response/parser mismatch: '+rid)
                    trace=OUT/study/'responses'/model/rid/'stdout.jsonl'
                    if not trace.exists():errors.append('Missing raw request trace: '+rid)
                    else:
                        events=[json.loads(line) for line in trace.read_text().splitlines() if line.strip().startswith('{')]
                        sessions=[e.get('thread_id') for e in events if e.get('type')=='thread.started']
                        messages=[e.get('item',{}) for e in events if e.get('type')=='item.completed']
                        answers=[e.get('text','').strip() for e in messages if e.get('type')=='agent_message']
                        tools=[e for e in messages if e.get('type') not in ['agent_message','reasoning','error']]
                        if sessions!=[response.get('session_id')] or not any(e.get('type')=='turn.completed' for e in events) or not answers or answers[-1]!=raw.strip() or tools:errors.append('Fresh-session/raw/tool trace mismatch: '+rid)
                    blind=pack/'blinded'/rid
                    if response['identity']['image_sha256']!=C.sha(blind/'image.png') or response['identity']['prompt_sha256']!=C.sha(blind/'prompt.txt'):errors.append('Commercial image/prompt hash mismatch: '+rid)
                iid=r['image_id']
                if iid in data:errors.append('Duplicate commercial pair: '+iid)
                data[iid]=dict(hit=records['adversarial']['parsed_label']==r['target_label'],clean_correct=records['clean']['parsed_label']==r['source_label'],clean_target=records['clean']['parsed_label']==r['target_label'],clean_valid=records['clean']['parsed_label'] is not None,adv_valid=records['adversarial']['parsed_label'] is not None,transition=r['transition'])
            key=(study,model);groups[key][pack.name]=data;expected={r['image_id'] for r in manifest.get('cases',[])};cohorts[key]=expected
            completion.append(dict(study=study,model=model,method=pack.name,observed=len(data),N=len(expected),status='COMPLETE' if len(data)==len(expected) else 'PARTIAL',hits=sum(x['hit'] for x in data.values())))
    stats=[];comparisons=[]
    for (study,target),methods in groups.items():
        expected=cohorts[(study,target)]
        for method,data in methods.items():
            for tid in ['ALL',*C.TIDS]:
                values=[v for v in data.values() if tid=='ALL' or v['transition']==tid];n=len(values);h=sum(v['hit'] for v in values)
                planned=len(expected) if tid=='ALL' else len(expected)//5;complete=n==planned
                lo,hi=wilson(h,n) if complete else (None,None)
                stats.append(dict(study=study,target=target,method=method,transition=tid,hits=h,N=planned,observed=n,ASR=h/planned if complete else None,Wilson95_low=lo,Wilson95_high=hi,
                    clean_correct=sum(v['clean_correct'] for v in values),clean_accuracy=sum(v['clean_correct'] for v in values)/planned if complete else None,clean_target_hits=sum(v['clean_target'] for v in values),new_target_hits=sum(v['hit'] and not v['clean_target'] for v in values),clean_parser_invalid=sum(not v['clean_valid'] for v in values),adv_parser_invalid=sum(not v['adv_valid'] for v in values),clean_correct_hits=sum(v['hit'] and v['clean_correct'] for v in values),
                    clean_correct_ASR=sum(v['hit'] and v['clean_correct'] for v in values)/sum(v['clean_correct'] for v in values) if complete and any(v['clean_correct'] for v in values) else None))
        if study!='commercial50':
            for a,b in itertools.combinations(sorted(methods),2):
                if set(methods[a])==expected and set(methods[b])==expected:
                    comparisons.append(dict(study=study,target=target,method_a=a,method_b=b,**paired(methods[a],methods[b])))
    # Holm family across the three final-method comparisons for each target.
    by=defaultdict(list)
    for row in comparisons:by[(row['study'],row['target'])].append(row)
    for rows in by.values():
        ordered=sorted(rows,key=lambda r:r['McNemar_exact_p']);prev=0
        for i,row in enumerate(ordered):prev=max(prev,min(1,(len(rows)-i)*row['McNemar_exact_p']));row['Holm_p_within_target']=prev
    audit=C.read(C.ROOT/'audits/model_dtype_audit.json',[]);bad=[v for v in audit if v.get('status')!='pass']
    if bad:errors.append('BF16 audit has non-pass events')
    refs=set()
    for path in (C.DATA/'manifests/class_references').glob('*.jsonl'):
        rr=C.rows(path)
        if len(rr)!=48:errors.append('Reference count mismatch: '+str(path))
        refs.update(r['image_id'] for r in rr)
    for split in ['explore','confirm','final150']:
        if refs & {r['image_id'] for r in C.cohort(split)['images']}:errors.append('Reference overlap: '+split)
    # Existing canonical manifest audit independently records old-cohort AND pixel overlap.
    cohort_audit=C.read(OUT/'final150_cohort_audit.json')
    write_csv(report/'paired_comparisons.csv',comparisons);write_csv(report/'confidence_intervals.csv',stats);write_csv(report/'completion_matrix.csv',completion);write_csv(report/'png_integrity.csv',receipts)
    C.atomic(report/'independent_validation.json',dict(at=time.time(),status='PASS' if not errors else 'FAIL',errors=errors,
      verified_attack_receipts=len(receipts),BF16_events=len(audit),BF16_nonpass=len(bad),planned_conditions=len(completion),completed_conditions=sum(r['status']=='COMPLETE' for r in completion),
      incomplete_conditions=sum(r['status']!='COMPLETE' for r in completion),source_script_sha256=C.sha(Path(__file__)),
      final_method_lock_sha256=C.sha(OUT/'final_method_lock.json'),cohort_audit=cohort_audit,
      limits=['BF16 audit verifies recorded loads; not unavailable closed-model weights','No seed replication or causal/equivalence proof','No representation geometry available for closed models']))
    text=['# Paired final150 interpretation','', 'Pure 1:1 Pull–Push, three legal proxies, four layers/model. GPT50 selected among predeclared candidates; final150 was evaluated after freeze.',
      'Tests use paired image outcomes, exact two-sided McNemar, Holm within target; 95% paired-bootstrap intervals stratify the five directions. No pooling of correlated target-image rows.','',
      '|Target|Comparison A vs B|Difference pp|95% CI pp|McNemar p|Holm p|','|---|---|---:|---|---:|---:|']
    for row in comparisons:
        if row['study'] in ['final150','commercial150']:
            text.append(f"|{row['target']}|{row['method_a']} vs {row['method_b']}|{row['difference_pp']:.2f}|[{row['CI95_low_pp']:.2f}, {row['CI95_high_pp']:.2f}]|{row['McNemar_exact_p']:.4g}|{row['Holm_p_within_target']:.4g}|")
    text+=['','Uniform4 remains a strong baseline. Differences of a few hits do not establish superiority; non-significance does not establish equivalence.',
      'FDA-inspired selection is an adaptation, not reproduction of auxiliary-classifier FDA. CKA/RSA are descriptive and cannot establish transfer mechanism alone.',
      'Interpret ratios only on their matched30 cohort; final150 and historical dev cohorts have different difficulty. Missing responses are unscored and never zeros.',
      '',f'Integrity checks: {"PASS" if not errors else "FAIL"}; incomplete conditions: {sum(r["status"]!="COMPLETE" for r in completion)}.']
    # A small contact sheet is display-only; retain the exact, unmodified input PNGs locally.
    examples=report/'examples';examples.mkdir(parents=True,exist_ok=True)
    if not (examples/'paired_examples.png').exists():
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        labels=['goldfish','monarch butterfly','pineapple','acoustic guitar','laptop','espresso','volcano','rocking chair','soccer ball','school bus']
        cases=[next(r for r in C.cohort('final150')['images'] if r['transition']==tid) for tid in C.TIDS]
        fig,axes=plt.subplots(5,4,figsize=(10,13));index=[]
        for row,r in enumerate(cases):
            methods=['uniform','selected','FDA'];first=final['families']['qwen']['uniform']['condition'];clean=C.read(C.image_receipt(first,r['image_id']))['clean']
            paths=[clean]+[C.read(C.image_receipt(final['families']['qwen'][m]['condition'],r['image_id']))['adv'] for m in methods]
            for col,(name,path) in enumerate(zip(['clean',*methods],paths)):
                axes[row,col].imshow(Image.open(path));axes[row,col].axis('off');axes[row,col].set_title(name,fontsize=10)
                index.append(dict(image_id=r['image_id'],transition=r['transition'],method=name,path=path,sha256=C.sha(path)))
            axes[row,0].text(0,-.07,labels[r['source']-1]+' → '+labels[r['target']-1],transform=axes[row,0].transAxes,fontsize=9)
        fig.suptitle('Same frozen150 images: clean and three methods; first image per direction',fontsize=12)
        fig.tight_layout(rect=(0,.01,1,.97));fig.savefig(examples/'paired_examples.png',dpi=150);plt.close(fig)
        C.atomic(examples/'index.json',dict(selection='first image per direction in frozen final150 manifest; not chosen by success',images=index,display_only=True))
    # Durable report uses human-readable method names and all planned rows.
    final_rows=[r for r in stats if r['study'] in ['final150','commercial150'] and r['transition']=='ALL']
    narrative=['# V8 final experiment results', '',f'Updated UTC: {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}',
        '', 'Status: '+('COMPLETE' if all(r['status']=='COMPLETE' for r in completion) and not errors else 'PARTIAL'),
        '', '## Protocol and selection', '',
        'Fixed 1:1 cosine-prototype Pull–Push; 48 disjoint references/class; deepest and intermediate native spatial-token mean pooling; four layers per proxy in the three frozen finalists. Each target uses CLIP plus two VLM proxies outside its entire target family. Equal per-image L1-normalized proxy gradient averaging; BF16 unquantized open weights; 50 steps; 224px; epsilon16/255; alpha1/255; momentum1; DI; deterministic seed42.',
        'The user authorized selecting selected4, FDA4 and uniform4 with seven predeclared GPT6.1-low 50-image development comparisons. This is commercial-feedback-assisted development, not untouched confirmation. Final150 has disjoint IDs/pixels and was not used for retuning. Generic commercial packets use the qwen-style CLIP/InternVL2B/GemmaE2B proxy pool.',
        'Selected4 uses independently ranked native layers from the historical ten-image development scan; FDA4 uses conditional greedy joint-loss additions. Both are adaptations evaluated here, not faithful FDA auxiliary-classifier reproduction. All final methods use three proxies and four layers/proxy, so these final rows do not demonstrate lower compute for selected4.',
        '', '## Final150 target success over all ground-truth-selected images', '',
        '|Target|Method|Hits / planned N|Observed pairs|ASR|Wilson 95% CI|','|---|---|---:|---:|---:|---|']
    for r in final_rows:
        interval=f"[{r['Wilson95_low']:.3f}, {r['Wilson95_high']:.3f}]" if r['ASR'] is not None else 'N/A'
        asr=f"{100*r['ASR']:.2f}%" if r['ASR'] is not None else 'N/A'
        narrative.append(f"|{r['target']}|{r['method']}|{r['hits']}/{r['N']}|{r['observed']}|{asr}|{interval}|")
    narrative+=['','## Clean baseline and parser diagnostics','','These diagnostics do not change image eligibility or the primary denominator. New targeted hits exclude images already predicted as the desired target when clean.','','|Target|Method|Clean correct / observed|Clean already-target|New targeted hits|Invalid clean / adv|','|---|---|---:|---:|---:|---:|']
    for r in final_rows:narrative.append(f"|{r['target']}|{r['method']}|{r['clean_correct']}/{r['observed']}|{r['clean_target_hits']}|{r['new_target_hits']}|{r['clean_parser_invalid']}/{r['adv_parser_invalid']}|")
    narrative+=['','## Direction breakdown','','|Target|Method|Direction|Hits / planned N|Observed|','|---|---|---|---:|---:|']
    for r in stats:
        if r['study'] in ['final150','commercial150'] and r['transition']!='ALL':narrative.append(f"|{r['target']}|{r['method']}|{r['transition']}|{r['hits']}/{r['N']}|{r['observed']}|")
    narrative+=['','## Additional matched30/50 held-out surrogate validation','','These development rows are method-selection evidence, not final-target confirmation.','','|Validator|Method|Split|Hits / N|Observed|','|---|---|---|---:|---:|']
    for r in stats:
        if r['study'].startswith('development_') and r['transition']=='ALL':narrative.append(f"|{r['target']}|{r['method']}|{r['study']}|{r['hits']}/{r['N']}|{r['observed']}|")
    narrative+=['','## Uniform4 Pull/Push ratio controls on matched30','','|Target|Pull|Push|Hits / N|Observed|','|---|---:|---:|---:|---:|']
    for r in stats:
        if r['study']=='ratio30' and r['transition']=='ALL':
            pull=float(r['method'].split('_')[-1]);narrative.append(f"|{r['target']}|{pull:g}|{1-pull:g}|{r['hits']}/{r['N']}|{r['observed']}|")
    narrative+=['','## Statistical interpretation','',*text[2:], '', '## Measurements and reproducibility', '',
        f'Independently verified {len(receipts)} unique frozen attack receipts and decoded PNG pixel bounds. Recorded BF16 loads: {len(audit)}; non-pass events: {len(bad)}. Full parameter requirements apply only to inspectable open weights; closed weights are unavailable.',
        f"Sum of per-image measured attack execution: {sum(r['seconds'] or 0 for r in receipts)/3600:.2f} worker-hours; backwards: {sum(r['backwards'] or 0 for r in receipts)}. Concurrent worker durations are not GPU walltime, and exclude offline selection, cache creation and failed attempts.",
        'Exact revisions and each native layer are in final_method_lock.json and all_BF16_load_audits.json.gz. results.csv/per_image.csv/per_direction.csv contain open target DeltaR, margins and cross-model CKA/RSA. confidence_intervals.csv, paired_comparisons.csv, completion_matrix.csv and png_integrity.csv provide independent audit outputs.',
        'projections/ exports joint PCA and t-SNE coordinates plus images for every available final target/method/direction. Each fit includes both reference classes plus clean/adversarial images; reverse attacks are absent. t-SNE distances are qualitative. CKA/RSA are means across proxy-layer/target reference geometry; individual values are exported. Closed-model DeltaR/CKA/RSA are N/A.',
        'examples/paired_examples.png displays the first frozen image per direction, with clean/Uniform/Selected/FDA images side by side; no success-based example selection. Exact image IDs and PNG hashes are in examples/index.json.',
        'Full PNGs, raw response traces, failed attempts and restartable checkpoints remain on the persistent root disk under '+str(C.ROOT)+'. GitHub carries code and compact complete numerical data, not the full image corpus.',
        '', '## Limits and outstanding work', '',
        'Only a fixed five-direction ten-class task and one primary seed were evaluated here. These results do not establish open-ended VQA, category generality, universal transfer or a causal mechanism. Small method differences are not automatically significant; shared images across models must not be treated as independent extra samples.',
        f"Incomplete planned conditions: {sum(r['status']!='COMPLETE' for r in completion)}. Missing results are N/A; no failed or incomplete response is counted as a complete150 run."]
    (report/'FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(narrative)+'\n')
    (report/'PAIRED_INTERPRETATION.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(errors=errors,conditions=len(completion),incomplete=sum(r['status']!='COMPLETE' for r in completion),paired_comparisons=len(comparisons),pngs=len(receipts))))
    return 1 if errors else 0
if __name__=='__main__':
    with (OUT/'statistics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raise SystemExit(run())
