"""Independent CPU recount of raw replies, original PNG pixels and GT cohorts."""
import hashlib
import json
import time
from collections import Counter
import numpy as np
from PIL import Image
from proxy_study_protocol import OUT, ROOT, EXP, records, sha, atomic_json, evaluator_hashes
from supplement_protocol import candidates


def parse(raw):
    line = next((s.strip() for s in raw.splitlines() if s.strip()), '')
    return int(line)+1 if line in tuple('0123456789') else None


def main():
    began=time.time(); dest=ROOT/'historical_audit'; dest.mkdir(parents=True,exist_ok=True)
    failures=[]; content={}; ids={}; refs=[]
    for label in range(1,11):
        bank=records(OUT/'manifests/class_references'/f'class_references_{label:02d}.jsonl')
        assert len(bank)==48 and all(x['human_label']==label for x in bank)
        refs+=bank
    groups={'reference':refs,'test':[],'dev':[]}
    for tid in ['T01','T03','T05','T07','T09']:
        test=records(OUT/'manifests/test'/f'{tid}.jsonl')
        pool=records(OUT.parents[0]/'a100_precommercial_v6/evaluation/manifests/transitions'/tid/'candidates.jsonl')
        expected=sorted(sorted(pool,key=lambda x:hashlib.sha256(f"42|{tid}|{x['image_id']}".encode()).hexdigest())[:30],key=lambda x:x['image_id'])
        assert [x['image_id'] for x in test]==[x['image_id'] for x in expected]
        assert all(x['human_label']==int(tid[1:]) for x in test)
        groups['test']+=test
        groups['dev']+=records(OUT/'manifests/dev'/f'{tid}.jsonl')
    for group,rows in groups.items():
        ids[group]={r['image_id'] for r in rows};content[group]=set()
        assert len(ids[group])==len(rows)
        for row in rows:
            assert row['synset'] in row['image_id']
            pixels=np.asarray(Image.open(OUT/'canonical_images'/row['relative_path']).convert('RGB'))
            assert pixels.shape==(224,224,3)
            content[group].add(hashlib.sha256(pixels.tobytes()).hexdigest())
    overlaps={}
    for a,b in [('reference','test'),('reference','dev'),('test','dev')]:
        overlaps[a+'__'+b]=dict(image_id_overlap=len(ids[a]&ids[b]),exact_pixel_overlap=len(content[a]&content[b]))
        assert not ids[a]&ids[b] and not content[a]&content[b]
    summary=[]; pngs=0; maxdelta=0; mismatches=0
    # Scan the predeclared original 47 conditions. Partials are reported, not treated as complete.
    for item in candidates():
        name=item['measurement_condition']; target=item['config']['target']; rows=[]; attackids=[]; dirs=Counter(); cleanhits=0; bad=0
        expected_by_tid={t:{x['image_id']:x for x in records(OUT/'manifests/test'/f'{t}.jsonl')} for t in ['T01','T03','T05','T07','T09']}
        for statepath in sorted((OUT/'attacks/test'/name).glob('T*/batch_*/state.json')):
            state=json.loads(statepath.read_text());tid=statepath.parent.parent.name
            assert state['target_access'] is False and state['steps']==50 and state['seed']==42 and state['restarts']==1
            assert state['source_label']==int(tid[1:]) and state['target_label']==int(tid[1:])+1
            for rel,h in state['png_hashes'].items():
                assert sha(OUT/rel)==h
            for iid,cp,ap in zip(state['image_ids'],state['clean_paths'],state['adv_paths']):
                ref=expected_by_tid[tid][iid]
                clean=np.asarray(Image.open(OUT/cp).convert('RGB'));adv=np.asarray(Image.open(OUT/ap).convert('RGB'))
                canonical=np.asarray(Image.open(OUT/'canonical_images'/ref['relative_path']).convert('RGB'))
                assert clean.shape==adv.shape==(224,224,3) and np.array_equal(clean,canonical)
                delta=int(np.abs(adv.astype('int16')-clean.astype('int16')).max());maxdelta=max(maxdelta,delta)
                assert 0<delta<=16
                pngs+=2;attackids.append((tid,iid))
            mp=OUT/'evaluations'/name/target.replace('/','_')/tid/statepath.parent.name/'metrics.json'
            if not mp.exists():
                continue
            m=json.loads(mp.read_text());assert [r['image_id'] for r in m['images']]==state['image_ids']
            for row in m['images']:
                clean=parse(row['clean_raw']);adv=parse(row['adv_raw']);hit=adv==state['target_label']
                if clean!=row['clean_label'] or adv!=row['adv_label'] or hit!=row['target_hit'] or (clean==state['source_label'])!=row['clean_correct']:
                    mismatches+=1
                cleanhits+=clean==state['target_label'];bad+=adv is None
                dirs[tid]+=1;rows.append(dict(transition=tid,image_id=row['image_id'],hit=hit,clean_correct=clean==state['source_label']))
        assert len(attackids)==len(set(attackids))
        assert len(rows)==len({(x['transition'],x['image_id']) for x in rows})
        complete=all(dirs[t]==30 for t in expected_by_tid)
        summary.append(dict(aliases=item['aliases'],condition=name,target=target,status='COMPLETE' if complete else 'PARTIAL' if attackids else 'NOT_STARTED',
                            attack_images=len(attackids),evaluated_images=len(rows),hits=sum(x['hit'] for x in rows),
                            denominator=len(rows),ASR=sum(x['hit'] for x in rows)/150 if complete else None,
                            clean_correct=sum(x['clean_correct'] for x in rows),clean_already_target=cleanhits,invalid_adv_outputs=bad))
        print('[independent audit]',item['aliases'],len(rows),sum(x['hit'] for x in rows),flush=True)
    events=[]
    for p in [OUT/'audits/model_dtype_audit.json',OUT/'multiproxy_8h_supplement/audits/model_dtype_audit.json']:
        if p.exists():
            events+=json.loads(p.read_text())
    failed=[e for e in events if e['status']!='pass' or not e['all_floating_parameters_bf16'] or set(e['parameter_counts_by_dtype'])!={'bfloat16'}]
    result=dict(status='PASS_WITH_CAVEATS' if mismatches==0 and not failed else 'NEEDS_REVISION',
                scope='Independent CPU raw-output recount and PNG/dataset checks; historical BF16 event review, not fresh model reload or serial GPU replay',
                started=began,finished=time.time(),complete_conditions=sum(s['status']=='COMPLETE' for s in summary),
                summaries=summary,pngs_checked=pngs,max_uint8_delta=maxdelta,parser_or_hit_mismatches=mismatches,
                data_groups={k:len(v) for k,v in groups.items()},overlaps=overlaps,
                bf16_events=len(events),failed_dtype_events=len(failed),evaluator_hashes_now=evaluator_hashes(),
                caveats=['Fixed ten-class forced-choice recognition, not open-ended QA.',
                         '30 images per direction; point estimates do not guarantee population ASR.',
                         'Existing evaluator cache keys omit evaluator/prompt/parser hashes; no demonstrated inflation. New study locks include them.',
                         'Historical cohort has been observed; further results are exploratory, requiring untouched confirmation.',
                         'Fresh B1 versus B2 target replay not performed by this CPU audit.'])
    atomic_json(dest/'independent_audit.json',result)
    from proxy_selection_tables import writecsv
    writecsv(dest/'recount.csv',[{**x,'aliases':'|'.join(x['aliases'])} for x in summary])
    (dest/'audit_report.md').write_text('# Independent historical result audit\n\n'+
        f"Status: {result['status']}. Complete conditions: {result['complete_conditions']}; PNGs: {pngs}; maximum pixel delta: {maxdelta}/255; parser/hit mismatches: {mismatches}; failed historical dtype events: {len(failed)}.\n\n"+
        '\n'.join('- '+x for x in result['caveats'])+'\n')
    if mismatches or failed:
        raise RuntimeError('Independent audit found mismatches; inspect report')


if __name__=='__main__':
    main()
