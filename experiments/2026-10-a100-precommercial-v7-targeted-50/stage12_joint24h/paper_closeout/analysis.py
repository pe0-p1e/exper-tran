"""CPU report/measurement export, complete raw numerical traces, no images/auth."""
import argparse,csv,fcntl,gzip,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as C
from paper_closeout.plan import OUT

def rate(hits,N,observed):return hits/N if N>0 and observed==N else None

def table(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r));tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def calculate():
    import numpy as np
    from geometry import reference,cka,rsa
    rows=[];directions=[];images=[];geometry=[];variance=[]
    for p in (OUT/'target_references').glob('*.json'):
        m=C.read(p)
        for i,s in enumerate(m['per_class'],1):variance.append(dict(model=m['model'],class_label=i,layer=m['layer'],N=48,**s))
    for p in (C.ROOT/'conditions').glob('*.json'):
        c=C.read(p)
        for target_dir in [C.ROOT/'validation',*[C.ROOT/'large_validation'/n for n in C.LARGE_EVAL_MODES.values()]]:
            ev=C.read(target_dir/(c['id']+'.json'))
            if not ev:continue
            target=ev['model'];meta=[]
            for x in (OUT/'target_embeddings'/C.digest(target)[:12]/c['id']).glob('*.json'):meta.append(C.read(x))
            by={x['image_id']:x for x in meta}
            for x in ev['images']:images.append(dict(condition=c['id'],model=target,split=c['split'],**x,**{k:by.get(x['image_id'],{}).get(k) for k in ['delta_pull','delta_push','DeltaR','clean_margin','adversarial_margin','margin_change','gap_closure','prototype_distance']}))
            cross=[]
            if meta:
                refs=next((C.read(p) for p in (OUT/'target_references').glob('*.json') if C.read(p)['model']==target),None)
                if refs:
                    npz=next(p.with_suffix('.npz') for p in (OUT/'target_references').glob('*.json') if C.read(p)['model']==target)
                    with np.load(npz) as a:y=a['z'].reshape(-1,a['z'].shape[-1])
                    for proxy,layers in c['layers'].items():
                        for layer in layers:
                            key=C.digest([C.REG[proxy],layer,C.REG[target],'same ordered480 references'])
                            cache=OUT/'cross_geometry_cache'/(key+'.json');item=C.read(cache)
                            if item is None:
                                x=reference(proxy,layer);item=dict(proxy=proxy,layer=layer,target=target,N=len(x),CKA=cka(x,y),RSA=rsa(x,y),sample_ids='same ordered10classes x48reference IDs')
                                C.atomic(cache,item)
                            item=dict(condition=c['id'],**item)
                            cross.append(item);geometry.append(item)
            def mean(key,values):
                vals=[r[key] for r in values if r.get(key) is not None];return float(np.mean(vals)) if vals else None
            row=dict(condition=c['id'],label=c['label'],split=c['split'],model=target,N=ev['N'],completed=ev.get('completed',len(ev['images'])),hits=ev['hits'],ASR=ev.get('ASR'),
              target_representation_shift=mean('DeltaR',meta),CKA=mean('CKA',cross),RSA=mean('RSA',cross),
              CKA_RSA_aggregation='descriptive mean across model-layer pairs; individual values exported',measured_images=len(meta),
              layers=json.dumps(c['layers']),pull=c['pull'],push=c['push'],metric='target hits/all GT-selected images')
            rows.append(row)
            for tid in C.TIDS:
                ix=[x for x in ev['images'] if x['transition']==tid];mm=[x for x in meta if x['transition']==tid]
                N=sum(x['transition']==tid for x in C.cohort(c['split'])['images'])
                directions.append(dict(condition=c['id'],model=target,transition=tid,hits=sum(x['target_hit'] for x in ix),N=N,observed=len(ix),ASR=rate(sum(x['target_hit'] for x in ix),N,len(ix)),
                  DeltaR=mean('DeltaR',mm),clean_margin=mean('clean_margin',mm),margin_change=mean('margin_change',mm),prototype_distance=mean('prototype_distance',mm),CKA=row['CKA'],RSA=row['RSA']))
    report=OUT/'reports';table(report/'results.csv',rows);table(report/'per_direction.csv',directions);table(report/'per_image.csv',images);table(report/'cross_model_CKA_RSA.csv',geometry);table(report/'class_variance.csv',variance)
    if len(directions)>=3:
        import pandas as pd
        df=pd.DataFrame(directions);cols=['ASR','DeltaR','clean_margin','margin_change','prototype_distance','CKA','RSA']
        corr=df[cols].corr(method='spearman');corr.to_csv(report/'spearman.csv');df[cols].notna().astype(int).T.dot(df[cols].notna().astype(int)).to_csv(report/'spearman_pairwise_N.csv')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,6));a=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(cols)),cols,rotation=60,ha='right');ax.set_yticks(range(len(cols)),cols);fig.colorbar(a,ax=ax);fig.tight_layout();fig.savefig(report/'spearman.png');plt.close(fig)
    from paper_closeout.commercial import summarize
    commercial={s:summarize(s) for s in ['commercial50','commercial150','ratio30']}
    text=['# Final experiment closeout','',f'Updated UTC: {time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())}',
      '', 'GPT50 is USED FOR SELECTION by explicit user instruction. Final150 is separate; final outcomes cannot change frozen settings.',
      'Models: GPT-6.1 Sol, GPT-6 Sol, GPT-5.6 Sol, all low; no Astra, no paid API fallback.',
      'Pure PP, BF16 unquantized,50steps,16/255,DI,equal per-image L1-normalized proxy gradients. References48/class. Ratios are separate Uniform4 controls.',
      '', '## Actual completed/partial open-weight evaluations','', '|Condition|Target|Hits / N|Complete|DeltaR|CKA|RSA|','|---|---|---|---|---|---|---|']
    for x in rows:
        if x['split']=='final150' or C.read(OUT/'activation.json',{}).get('include_all_history',False):text.append('|'+ '|'.join(str(v) for v in [x['condition'],x['model'],f"{x['hits']}/{x['N']}",x['ASR'] is not None,x['target_representation_shift'],x['CKA'],x['RSA']])+'|')
    for study,result in commercial.items():
        text+=['',f'## {study}: '+('COMPLETE' if result['complete'] else 'PARTIAL')]
        for x in result['methods']:text.append(f"- {x['model']} / {x['method']}: {x['hits']} hits, {x['completed_pairs']}/{x['N']} completed; ASR={x['ASR']}")
    text+=['','Missing measurements remain N/A, never zero. Closed-model target DeltaR/CKA/RSA are unavailable. Correlations are exploratory and correlated across shared images/models. No causality claim.',
      'PCA/t-SNE are fitted jointly per model/class pair; missing reverse-direction attacks are explicitly absent, not fabricated. No new reverse attack campaign is claimed.',
      'Complete numerical receipts, gradient traces, configurations and image-level results are in the uploaded compressed data bundle; full PNGs and raw CLI traces stay on the persistent root disk.']
    (report/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
    return rows

def projections():
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs={}
    for p in (OUT/'target_references').glob('*.json'):
        meta=C.read(p)
        with np.load(p.with_suffix('.npz')) as d:refs[C.digest(meta['model'])[:12]]=d['z']
    for model_dir in (OUT/'target_embeddings').glob('*'):
        if model_dir.name not in refs:continue
        for condition in model_dir.glob('*'):
            c=C.read(C.ROOT/'conditions'/(condition.name+'.json'))
            if not c or c['split']!='final150':continue
            for tid in C.TIDS:
                dest=OUT/'reports/projections'/model_dir.name/condition.name/tid
                if (dest/'complete.json').exists():continue
                cases=[r for r in C.cohort('final150')['images'] if r['transition']==tid];files=[condition/(C.digest(r['image_id'])[:18]+'.npz') for r in cases]
                if not all(p.exists() for p in files):continue
                clean=[];adv=[]
                for p in files:
                    with np.load(p) as d:clean.append(d['clean']);adv.append(d['adv'])
                source,target=cases[0]['source']-1,cases[0]['target']-1;bank=refs[model_dir.name]
                X=np.concatenate([bank[source],bank[target],np.stack(clean),np.stack(adv)]).astype(np.float64);X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
                groups=['source_ref']*48+['target_ref']*48+['clean']*30+['adv']*30
                pre=PCA(n_components=min(40,X.shape[1],len(X)-1),random_state=42).fit_transform(X)
                for name,z in [('PCA',PCA(n_components=2,random_state=42).fit_transform(X)),('tSNE',TSNE(n_components=2,random_state=42,init='pca',perplexity=30).fit_transform(pre))]:
                    table(dest/(name+'.csv'),[dict(index=i,group=g,x=float(v[0]),y=float(v[1])) for i,(g,v) in enumerate(zip(groups,z))])
                    fig,ax=plt.subplots()
                    for group in ['source_ref','target_ref','clean','adv']:
                        ix=[i for i,g in enumerate(groups) if g==group];ax.scatter(z[ix,0],z[ix,1],s=12,label=group,alpha=.7)
                    ax.legend();ax.set_title(name+' joint fit; forward direction only');fig.tight_layout();fig.savefig(dest/(name+'.png'));plt.close(fig)
                C.atomic(dest/'complete.json',dict(joint_fit=True,normalization='L2',random_state=42,perplexity=30,N=len(X),reverse_direction_present=False))

def publish():
    """All numeric receipts, chunked below GitHub limits, plus compact old reports."""
    dest=C.EXP/'reports/v8_paper_closeout';dest.mkdir(parents=True,exist_ok=True)
    for sub in ['reports','selection_locks','commercial50','commercial150','ratio30']:
        for p in (OUT/sub).rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ['blinded','responses','logs','private']):continue
            if p.suffix not in ['.json','.csv','.md','.png']:continue
            q=dest/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for name in ['final_method_lock.json','pull_push_ablation_lock.json','activation.json','final150_cohort_audit.json','commercial_preflight.json']:
        if (OUT/name).exists():shutil.copyfile(OUT/name,dest/name)
    # Full per-step traces, not just their means; no PNG content/model weights/auth.
    bundle=dest/'numerical_receipts';bundle.mkdir(exist_ok=True);manifest=[];batch=[];index=0
    def flush():
        nonlocal batch,index
        if not batch:return
        q=bundle/f'part_{index:04d}.jsonl.gz'
        with q.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
                for x in batch:f.write((json.dumps(x,separators=(',',':'))+'\n').encode())
        manifest.append(dict(file=str(q.relative_to(dest)),sha256=C.sha(q),records=len(batch)));batch=[];index+=1
    paths=sorted((C.ROOT/'attacks').glob('*/*/state.json'))+sorted((OUT/'target_embeddings').glob('*/*/*.json'))
    for p in paths:
        batch.append(dict(source=str(p.relative_to(C.ROOT)),sha256=C.sha(p),value=C.read(p)))
        if len(batch)>=200:flush()
    flush();C.atomic(dest/'data_manifest.json',dict(chunks=manifest,all_available_attack_receipts=True,all_available_target_measurements=True,PNG_upload=False))
    # Historical hyperparameter settings/individual outcomes remain separately labeled.
    for folder in ['pull_push_composition_ablation','pull_push_four_layer','pull_push_di_removal','pull_push_mechanism_pilots','ten_hour_research','family_disjoint_recovery']:
        root=C.DATA/folder
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix not in ['.csv','.md','.yaml']:continue
            if not any(x in p.parts for x in ['reports','results']) and 'ablation' not in p.name:continue
            q=dest/'historical'/folder/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
            if p.stat().st_size<30*2**20:shutil.copyfile(p,q)
            else:
                with p.open('rb') as src,gzip.open(str(q)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    # Raw embeddings are numerical intermediates, each small NPZ is Git-friendly.
    for base in [OUT/'target_embeddings',OUT/'target_references']:
        for p in base.rglob('*.npz'):
            q=dest/'embeddings'/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for p in (C.ROOT/'attacks').glob('*/*/features.npz'):
        q=dest/'embeddings/source'/p.relative_to(C.ROOT/'attacks');q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for pattern in ['manifests/*.json','conditions/*.json','selection_geometry/*.json','geometry/*.json','fda_matched/*.json','stage1_methods/*policy.json','stage1_methods/*method_results.json']:
        for p in C.ROOT.glob(pattern):
            q=dest/'provenance'/p.relative_to(C.ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    audit=C.ROOT/'audits/model_dtype_audit.json'
    if audit.exists():
        values=C.read(audit)
        with gzip.open(dest/'all_BF16_load_audits.json.gz','wt') as f:json.dump(values,f)
        C.atomic(dest/'BF16_summary.json',dict(events=len(values),all_pass=all(v.get('status')=='pass' for v in values)))
    lock=C.DATA/'proxy_composition_18h/publication.lock'
    with lock.open('a') as h:
        fcntl.flock(h,fcntl.LOCK_EX)
        rel=str(dest.relative_to(C.REPO));subprocess.run(['git','add',rel],cwd=C.REPO,check=True)
        if subprocess.run(['git','diff','--cached','--quiet','--',rel],cwd=C.REPO).returncode:
            subprocess.run(['git','commit','-m','Publish complete compact closeout measurements and historical hyperparameter data'],cwd=C.REPO,check=True)
        subprocess.run(['git','push','git@github.com:pe0-p1e/exper-tran.git','HEAD:experiment/a100-precommercial-v8-joint-selection-24h'],cwd=C.REPO,check=True)

def main(deadline):
    stop=False
    def handler(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    last_publish=0
    while True:
        try:
            calculate();projections()
            if time.time()-last_publish>900 or stop or C.deadline_reached(deadline,3600):
                publish();last_publish=time.time()
            C.atomic(OUT/'analysis_heartbeat.json',dict(at=time.time(),last_publish=last_publish))
        except Exception as e:
            import traceback
            C.atomic(OUT/'analysis_error.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc()))
        finished=C.read(C.ROOT/'scheduler.json',{}).get('state')=='COMPLETE'
        if stop or finished or C.deadline_reached(deadline,1800):
            # A stop signal can precede the final atomic worker receipts. Drain only
            # verified own leases, then recalculate/export those last measurements.
            until=min(deadline,time.time()+600) if C.timebox_enabled() else time.time()+600
            while time.time()<until:
                own=C.read(C.ROOT/'scheduler.json',{}).get('active',[])
                if not any(C.same_process(j['process_identity']) for j in own):break
                time.sleep(3)
            try:calculate();projections()
            except Exception as e:C.atomic(OUT/'final_analysis_error.json',dict(error=str(e),at=time.time()))
            try:publish()
            except Exception as e:C.atomic(OUT/'publication_error.json',dict(error=str(e),at=time.time()))
            C.atomic(OUT/'analysis_driver_result.json',dict(status='CHECKPOINTED',at=time.time()));return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);a=p.parse_args();os.nice(10);main(a.deadline)
