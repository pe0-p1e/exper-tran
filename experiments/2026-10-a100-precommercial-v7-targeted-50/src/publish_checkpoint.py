"""Publish a bounded scientific checkpoint while full artifacts stay on the root disk."""
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from proxy_study_protocol import OUT, ROOT, EXP, atomic_json
from archive_proxy_study import archive as archive_composition
from archive_unified import archive as archive_measurements


def stable_bytes(p):
    for _ in range(3):
        before=p.stat();data=p.read_bytes();after=p.stat()
        if (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns):return data
    raise RuntimeError('Source is changing; snapshot not published: '+str(p))


def main():
    repo=EXP.parents[1]
    with (ROOT/'publication.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        dest=EXP/'reports/checkpoint_20261020';dest.mkdir(parents=True,exist_ok=True)
        local=ROOT/'snapshots'/stamp;local.mkdir(parents=True)
        archive_composition(False);archive_measurements(False)
        patterns={
            'evaluations':['**/metrics.json'],
            'development_evaluations':['**/metrics.json'],
            'manifests':['**/*.json','**/*.jsonl','**/*.csv'],
            'audits':['*.json'],
            'family_disjoint_recovery':['method_lock.json','current_stage.json','reports/*.csv','reports/*.md','results/*.csv','configs/*.json'],
            'multiproxy_8h_supplement':['method_lock.json','results/*.csv','reports/*.csv','reports/*.md','audits/*.json'],
            'proxy_selection':['*.csv','*.json','reports/*.csv','reports/*.md','results/*.csv'],
            'ten_hour_research':['*.csv','*.json','*.md','analysis/*.csv'],
            'analysis':['**/*.csv','**/*.json','**/*.md'],
            'proxy_composition_18h':['configs/*.json','calibration/*.json','audits/*.json','surrogate_followup/configs/*.json',
                'commercial_model_versions/resumed_gpt_state.json','commercial_model_versions/resumed_gpt_unhandled_error.json']}
        copied=[]
        for folder,globs in patterns.items():
            for pattern in globs:
                for p in (OUT/folder).glob(pattern):
                    if not p.is_file():continue
                    rel=p.relative_to(OUT);target=dest/'data'/rel
                    if str(rel) in {x['source'] for x in copied}:continue
                    data=stable_bytes(p);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                    copied.append(dict(source=str(rel),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        # Complete state settings, IDs, image hashes and compute counters, excluding large step traces.
        batches=dest/'attack_batch_metadata.jsonl';batch_count=0
        with batches.open('w') as f:
            for p in sorted((OUT/'attacks').glob('*/*/T*/batch_*/state.json')):
                raw=stable_bytes(p);d=json.loads(raw);trace=d.pop('trace',None)
                d.update(source_state=str(p.relative_to(OUT)),source_sha256=hashlib.sha256(raw).hexdigest(),
                         optimization_trace_retained_locally=trace is not None)
                f.write(json.dumps(d,allow_nan=False)+'\n');batch_count+=1
        # Inspect all local output entries without following directory links or touching model caches.
        inventory=local/'local_result_inventory.csv';counts={'files':0,'bytes':0,'symlinks':0};external=[]
        with inventory.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['path','bytes','mtime_ns','device','resolved','symlink']);w.writeheader()
            for current,dirs,files in os.walk(OUT,followlinks=False):
                dirs[:]=[d for d in dirs if Path(current)/d!=ROOT/'snapshots']
                for name in files+[d for d in dirs if (Path(current)/d).is_symlink()]:
                    p=Path(current)/name
                    if p==inventory:continue
                    try:s=p.stat();resolved=p.resolve()
                    except FileNotFoundError:external.append(dict(path=str(p),reason='broken link'));continue
                    link=p.is_symlink()
                    w.writerow(dict(path=str(p.relative_to(OUT)),bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                                    device=s.st_dev,resolved=str(resolved),symlink=link))
                    counts['files']+=1;counts['bytes']+=s.st_size;counts['symlinks']+=link
                    if resolved.is_relative_to(Path('/ephemeral')):external.append(dict(path=str(p),resolved=str(resolved)))
        storage=dict(at=stamp,output_root=str(OUT),resolved_root=str(OUT.resolve()),root_device=OUT.stat().st_dev,
                     root_mount=subprocess.check_output(['findmnt','-n','-o','SOURCE,TARGET','-T',str(OUT)],text=True).strip(),
                     local_inventory=str(inventory),ephemeral_output_references=external,**counts,
                     model_caches_excluded=True,live_campaign_snapshot=True)
        atomic_json(dest/'storage_audit.json',storage)
        if external:raise RuntimeError('Output links require persistent-disk repair before publication')
        atomic_json(dest/'snapshot_manifest.json',dict(at=stamp,files=copied,attack_batches=batch_count,
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    collection_is_live=True,missing_results_are_not_zeros=True))
        composition=list(csv.DictReader((EXP/'reports/proxy_study_18h/reports/completion.csv').open()))
        status={s:sum(r['status']==s for r in composition) for s in sorted({r['status'] for r in composition})}
        research=json.loads((EXP/'reports/unified_campaign/surrogate_followup/progress.json').read_text())
        commercial=json.loads((EXP/'reports/unified_campaign/commercial_model_versions/stages/test150/reports/summary.json').read_text())
        lines=['# Direction differences persist while the broader composition campaign remains incomplete','',
            f'Checkpoint collected at {stamp} UTC. This is a live partial-campaign snapshot; it does not mark pending work as finished.','',
            '## Coverage','',f'Composition conditions: {json.dumps(status)}. Total declared rows: {len(composition)}.',
            f'Follow-up controlled conditions: {sum(c["status"]=="COMPLETE" for c in research["conditions"])}/{len(research["conditions"])} complete.',
            'Full per-condition, per-direction and paired comparisons: [composition reports](../proxy_study_18h/reports/).',
            'Detailed frozen target batches, manifests, settings and BF16 audits: [snapshot data](data/).',
            'Attack batch identities, layer settings, counters and PNG hashes: [metadata](attack_batch_metadata.jsonl).','',
            '## Commercial evaluation','',
            '| Requested model | Returned adversarial hits / observed images | Full cohort status |',
            '|---|---:|---|']
        for model in commercial['models']:
            # Count every returned adversarial response, including images lacking a clean companion.
            spec=json.loads((EXP/'commercial_model_matrix.json').read_text())['models']
            s=next(x for x in spec if x['id']==model['matrix_id'])
            pack=json.loads((OUT/'commercial_test150/private/manifest.json').read_text())
            folder=ROOT/'commercial_model_versions/stages/test150/commercial_cli'/s['provider']/s['model']
            hits=n=0
            for c in pack['cases']:
                p=folder/c['requests']['adversarial']/'result.json'
                if p.exists():
                    d=json.loads(stable_bytes(p));n+=1;hits+=d['parsed_label']==c['target_label']
            lines.append(f'| {s["model"]} | {hits}/{n} | {"COMPLETE" if n==150 else "PARTIAL / no full ASR"} |')
        lines += ['', 'Returned-image rates on incomplete cohorts are descriptive partial observations. They must not be treated as full-cohort ASR. Gemini entries are reasoning profiles of one generation. CLI IDs do not establish undisclosed backend snapshots.', '',
            '## Scientific settings and interpretation','',
            'The composition study uses A3: four visual depths (25/50/75/100%), ten-class prototype margin, local patch weight 0.35, DI, and equal normalized proxy gradients. Fixed GT-selected 5×30 images, 48 disjoint reference images/class, seed42, 50 steps, L∞16/255, step1/255 and momentum1. All participating open-weight model parameters are BF16, without quantization. Targets evaluate frozen PNGs only.',
            'The follow-up loss/fusion controls use a separate ten-image surrogate development cohort. Their results are not final-target test scores. Commercial feedback does not select loss, proxies or attack parameters.',
            'Direction differences are the main descriptive finding. Reverse-direction measurements are needed before claiming directional asymmetry. Reference CKA/RSA, class dispersion, gradient conflicts, ΔR and margins are associations rather than causal explanations. PCA/t-SNE groups use a shared fit; t-SNE is qualitative.', '',
            '## Measurement artifacts','',
            '[Reference geometry](../unified_campaign/mechanism_reference/), [gradient compatibility](../unified_campaign/mechanism_gradients/), [representation measurements and joint coordinates](../unified_campaign/surrogate_followup/analysis/), [commercial per-request provenance](../unified_campaign/commercial_request_provenance.json).','',
            '## Storage and remaining work','',
            f'Full local artifacts remain at `{OUT}` on `{storage["root_mount"]}`. The storage audit found zero output links into `/ephemeral`. PNGs, NPZ feature arrays, complete optimization traces, raw CLI traces and resumable states remain local. OAuth credentials are outside this publication.',
            f'Local inventory: `{inventory}`. Model weight caches may remain on ephemeral storage. Root persistence covers ordinary hibernation/restart while the persistent disk is retained; VM/disk deletion is not an independent backup.',
            'Running jobs are not paused by this checkpoint. Pending composition cells, incomplete follow-up controls, seeds and blocked commercial calls retain their explicit status. No new inference was launched by the snapshot exporter.', '']
        (dest/'REPORT.md').write_text('\n'.join(lines))
        # Local compressed metadata backup; large images/features already remain on the persistent output disk.
        bundle=local/'scientific_metadata.tar.gz'
        with tarfile.open(bundle,'w:gz') as tar:
            for folder in [dest,EXP/'reports/unified_campaign',EXP/'reports/proxy_study_18h']:
                tar.add(folder,arcname=folder.name)
        atomic_json(local/'backup_receipt.json',dict(at=stamp,archive=str(bundle),bytes=bundle.stat().st_size,
                     sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),full_artifacts=str(OUT),storage_audit=storage))
        subprocess.run(['git','add',str(EXP.relative_to(repo)/'reports')],cwd=repo,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
            subprocess.run(['git','commit','-m','Publish current scientific checkpoint with partial results and persistent-storage audit'],cwd=repo,check=True)
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git',
                        'HEAD:experiment/a100-precommercial-v7-proxy-study-18h'],cwd=repo,check=True)
        print('Published checkpoint',dest,'Local metadata backup',bundle,flush=True)


if __name__=='__main__':main()
