#!/usr/bin/env python3
"""CPU-only immutable Stage1 snapshot, recomputed from per-image evidence."""
import csv,hashlib,json,math,os,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from primary_ml_cka.prompts.parser import parse_exact_label

def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def mdtable(rows,keys):
    def cell(v):
        if v is None:return 'N/A'
        if isinstance(v,(dict,list)):v=json.dumps(v,ensure_ascii=False)
        return str(v).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+['| '+' | '.join(cell(r.get(k)) for k in keys)+' |' for r in rows])

def wilson(h,n):
    if not n:return None,None
    z=1.95996398454;p=h/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-delta,center+delta

def main():
    cutoff=time.time();stamp=datetime.fromtimestamp(cutoff,timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=C.ROOT/'stage1_summary'/stamp;out.mkdir(parents=True,exist_ok=False)
    evidence={};issues=[];cohorts={s:C.cohort(s) for s in ['smoke','screen','explore','confirm']}
    def read(p):
        p=Path(p)
        if not p.exists() or p.stat().st_mtime>cutoff:return None
        raw=p.read_bytes();evidence[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    def live_read(p):
        # Aggregate audit/state files are atomically refreshed by running workers.
        # Preserve exact input bytes; filter individual audit timestamps below.
        p=Path(p)
        if not p.exists():return None
        raw=p.read_bytes();evidence[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    scheduler=live_read(C.ROOT/'scheduler.json') or {}
    audit_input=live_read(C.ROOT/'audits/model_dtype_audit.json') or []
    audits=[x for x in audit_input if datetime.fromisoformat(x['audited_utc']).timestamp()<=cutoff]
    conditions={p.stem:read(p) for p in sorted((C.ROOT/'conditions').glob('*.json'))};conditions={k:v for k,v in conditions.items() if v}
    registry=read(C.ROOT/'stage1_methods/registry.json') or {}
    aliases={k:v['methods'] for k,v in registry.items()}
    # Include declared follow-ups even when not admitted yet.
    for fn in ['selected_followup/method_lock.json','fda_followup/method_lock.json']:
        lock=read(C.ROOT/fn) or {}
        for value in lock.get('families',{}).values():
            for c in value.values():
                if isinstance(c,dict) and 'id' in c:conditions.setdefault(c['id'],c)
    allrows=[];perimage=[];direction=[];validated={};pngs={};receiptrows=[]
    def eval_record(c,p,role):
        ev=read(p)
        if ev is None:return None
        expected={x['image_id']:x for x in cohorts[c['split']]['images']};rows=ev.get('images',[]);ids=[x['image_id'] for x in rows]
        if len(ids)!=len(set(ids)) or set(ids)-expected.keys():issues.append(dict(path=str(p),error='duplicate or wrong cohort IDs'))
        for x in rows:
            if any(x.get(k)!=expected[x['image_id']][k] for k in ['source','target','transition']):issues.append(dict(path=str(p),error='wrong labels'))
            if x.get('target_hit')!=(x.get('adv_label')==x['target']):issues.append(dict(path=str(p),error='stored target_hit differs from labels'))
            if 'adv_raw' in x and parse_exact_label(x['adv_raw']).label!=x.get('adv_label'):issues.append(dict(path=str(p),error='raw parser mismatch'))
        complete=bool(ev.get('complete') and set(ids)==expected.keys() and len(ids)==len(expected))
        n=len(expected);hits=sum(x.get('adv_label')==x['target'] for x in rows);lo,hi=wilson(hits,n) if complete else (None,None)
        if ev.get('complete') and (ev.get('N')!=n or ev.get('hits')!=hits):issues.append(dict(path=str(p),error='N or numerator mismatch'))
        good=sum(x.get('clean_correct',False) for x in rows);conditional=sum(x.get('clean_correct',False) and x.get('adv_label')==x['target'] for x in rows)
        row=dict(condition=c['id'],family=c['family'],split=c['split'],label=c['label'],role=role,model=ev.get('model'),status='COMPLETE' if complete else 'PARTIAL',
          N=n,observed=len(rows),hits=hits if complete else None,ASR=hits/n if complete else None,CI95_low=lo,CI95_high=hi,
          clean_correct=good,clean_correct_targeted_hits=conditional,clean_correct_conditional_ASR=conditional/good if good and complete else None,
          invalid_parser=sum(x.get('parser_status')!='ok' for x in rows),layers=c['layers'],proxies=len(c['layers']),taps=sum(map(len,c['layers'].values())),
          evidence_path=str(p),evidence_sha256=evidence[str(p)],seconds=ev.get('seconds'),peak_reserved_gib=ev.get('peak_reserved_gib'))
        for x in rows:perimage.append(dict(condition=c['id'],role=role,model=ev.get('model'),split=c['split'],**x))
        for t in C.TIDS:
            rr=[x for x in rows if x['transition']==t];N=sum(x['transition']==t for x in expected.values())
            direction.append(dict(condition=c['id'],family=c['family'],role=role,model=ev.get('model'),split=c['split'],transition=t,
              observed=len(rr),N=N,hits=sum(x.get('adv_label')==x['target'] for x in rr) if len(rr)==N else None))
        validated[(c['id'],ev.get('model'))]=ev
        return row
    for cid,c in conditions.items():
        C.legal(c['layers'],c['validator']);attack_n=0
        for image in cohorts[c['split']]['images']:
            receipt=read(C.image_receipt(c,image['image_id']))
            if not receipt or receipt.get('status')!='COMPLETE':continue
            if receipt.get('science')!=C.digest(C.scientific(c)) or receipt.get('target_access') is not False:issues.append(dict(condition=cid,error='receipt science/target access'))
            if any(receipt.get(k)!=image[k] for k in ['image_id','source','target','transition']):issues.append(dict(condition=cid,error='receipt image label mismatch'))
            attack_n+=1
            for p,h in receipt.get('png_hashes',{}).items():pngs[p]=h
            receiptrows.append(dict(condition=cid,image_id=image['image_id'],split=c['split'],steps=receipt['steps'],target_access=receipt['target_access'],
              clean=receipt['clean'],adv=receipt['adv'],linf=receipt['linf_png'],seconds=receipt['seconds'],peak_reserved_gib=receipt['peak_reserved_gib'],backwards=receipt['model_backward_calls']))
        value=eval_record(c,C.eval_path(c),'DEV_VALIDATOR')
        if value:value['attack_images']=attack_n;value['method_aliases']=aliases.get(cid,[]);allrows.append(value)
        else:allrows.append(dict(condition=cid,family=c['family'],split=c['split'],label=c['label'],role='DEV_VALIDATOR',model=c['validator'],status='PARTIAL' if attack_n else 'QUEUED',N=len(cohorts[c['split']]['images']),observed=0,hits=None,ASR=None,layers=c['layers'],proxies=len(c['layers']),taps=sum(map(len,c['layers'].values())),attack_images=attack_n,method_aliases=aliases.get(cid,[])))
        if c['split']=='confirm':
            for mode,name in C.LARGE_EVAL_MODES.items():
                if C.REG[name]['family']!=c['family']:continue
                C.legal(c['layers'],name)
                p=C.ROOT/'large_validation'/name/(cid+'.json');value=eval_record(c,p,'LARGE_TARGET_CONFIRMATION')
                if value:allrows.append(value)
    # Independently verify saved PNG hashes and actual uint8 L-infinity bounds.
    from PIL import Image
    import numpy as np
    bad_hashes=[p for p,h in pngs.items() if not Path(p).exists() or C.sha(p)!=h]
    checked_pairs=set();bad_bounds=[]
    for x in receiptrows:
        pair=(x['clean'],x['adv'])
        if pair in checked_pairs:continue
        checked_pairs.add(pair)
        with Image.open(pair[0]) as a,Image.open(pair[1]) as b:
            aa=np.asarray(a.convert('RGB'),dtype=np.int16);bb=np.asarray(b.convert('RGB'),dtype=np.int16)
        maximum=int(np.abs(aa-bb).max()) if aa.shape==bb.shape else 256
        if maximum>16:bad_bounds.append(dict(condition=x['condition'],image_id=x['image_id'],max_uint8_delta=maximum))
    bad_audits=[dict(model=x.get('model_name'),role=x.get('role'),status=x.get('status')) for x in audits if not x.get('all_floating_parameters_bf16') or x.get('status')!='pass' or x.get('quantization',{}).get('quantized_modules')]
    # Condition-level source measurements: avoid treating steps/layers as independent samples.
    gs=defaultdict(list);rs=defaultdict(list);measurement_files=[]
    for fn,target,key in [('gradient_compatibility.csv',gs,'cos'),('representation_shift_variance.csv',rs,'DeltaR')]:
        p=C.ROOT/'reports'/fn
        if p.exists():
            with p.open() as f:
                for x in csv.DictReader(f):
                    try:target[x['condition']].append(float(x[key]))
                    except (KeyError,ValueError):pass
            measurement_files.append(dict(file=str(p),sha256=C.sha(p),mtime_UTC=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),rows=sum(map(len,target.values()))))
    mechanism=[dict(condition=cid,gradient_cos_mean=sum(gs[cid])/len(gs[cid]) if gs[cid] else None,gradient_observations=len(gs[cid]),DeltaR_mean=sum(rs[cid])/len(rs[cid]) if rs[cid] else None,representation_observations=len(rs[cid])) for cid in set(gs)|set(rs)]
    csvout(out/'all_conditions.csv',allrows);csvout(out/'per_image_evaluations.csv',perimage);csvout(out/'per_direction.csv',direction);csvout(out/'attack_receipts.csv',receiptrows);csvout(out/'mechanism_summary.csv',mechanism)
    aliases_rows=[];methods=[]
    for fam in C.VALIDATORS:
        result=read(C.ROOT/'stage1_methods'/(fam+'_method_results.json')) or {}
        for method in ['greedy','beam','fixed_proxy_best','independent','PRAF']:
            ids=result.get(method,[]);ids=ids if isinstance(ids,list) else [ids]
            scores=[]
            for cid in ids:
                if cid not in conditions:continue
                c=conditions[cid];ev=validated.get((cid,c['validator']))
                row=dict(family=fam,method=method,condition=cid,hits=ev.get('hits') if ev else None,N=10,layers=c['layers'],proxies=len(c['layers']),taps=sum(map(len,c['layers'].values())))
                aliases_rows.append(row)
                if ev and ev.get('complete'):scores.append(row)
            if scores:
                best=max(scores,key=lambda x:x['hits']);methods.append(dict(family=fam,method=method,hits_N=f"{best['hits']}/{best['N']}",condition=best['condition'],layers=best['layers']))
    csvout(out/'paper_method_candidates.csv',aliases_rows)
    stage1=read(C.ROOT/'stage1_methods/status.json') or {}
    layers=read(C.ROOT/'reports/three_proxy_selected_layers.json') or {};positions=read(C.ROOT/'reports/four_layer_positions.json') or {}
    csvout(out/'three_proxy_layer_results.csv',layers.get('rows',[]));csvout(out/'four_layer_position_results.csv',positions.get('rows',[]))
    count_rows=[]
    for fam in C.VALIDATORS:
        for method in sorted({m for v in registry.values() if v['condition']['family']==fam for m in v['methods']}):
            entries=[v for v in registry.values() if v['condition']['family']==fam and method in v['methods']]
            count_rows.append(dict(family=fam,method=method,complete=sum(bool(validated.get((v['condition']['id'],v['condition']['validator']),{}).get('complete')) for v in entries),planned=len(entries)))
    csvout(out/'method_completion.csv',count_rows)
    confirmation=[x for x in allrows if x['split']=='confirm'];csvout(out/'confirmation50.csv',confirmation)
    commercial=read(C.ROOT/'stage2/summary.json') or {};follow_commercial=read(C.ROOT/'fda_selected_commercial/summary.json')
    for name,source in [('main_method_lock.json',C.ROOT/'method_lock.json'),('selected_method_lock.json',C.ROOT/'selected_followup/method_lock.json'),('FDA_method_lock.json',C.ROOT/'fda_followup/method_lock.json'),('commercial_comparison_lock.json',C.ROOT/'fda_selected_commercial/method_lock.json')]:
        value=read(source)
        if value is not None:C.atomic(out/name,value)
    validation=dict(saved_png_hashes_checked=len(pngs),bad_hashes=bad_hashes,actual_uint8_linf_pairs_checked=len(checked_pairs),bad_bounds=bad_bounds,
      parameter_audit_events=len(audits),failed_BF16_or_quantization_events=bad_audits,evaluation_reconciliation_issues=issues,
      cutoff_UTC=datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),result='PASS' if audits and not bad_hashes and not bad_bounds and not bad_audits and not issues else 'NEEDS_REVIEW')
    C.atomic(out/'validation.json',validation)
    snapshot=dict(as_of_UTC=validation['cutoff_UTC'],state=scheduler.get('state'),stage1_methods=stage1,counts=dict(unique_conditions=len(conditions),evaluation_rows=len(allrows),per_image_rows=len(perimage),attack_receipts=len(receiptrows)),
      method_scores=methods,conditions=allrows,layer_results=layers.get('rows',[]),position_results=positions.get('rows',[]),validation=validation,
      commercial_baseline=commercial,commercial_followup=follow_commercial,measurement_sources=measurement_files,
      source_manifest=evidence,script_sha256=C.sha(__file__))
    C.atomic(out/'snapshot.json',snapshot)
    n30=[dict(family=x['family'],method=x['strategy'],hits_N=f"{x['hits']}/{x['N']}" if x['complete'] else 'PENDING',paired=x.get('paired_vs_uniform')) for x in layers.get('rows',[]) if x['split']=='explore']
    c50=[dict(model=x['model'],method=x['label'],hits_N=f"{x['hits']}/{x['N']}" if x['status']=='COMPLETE' else 'PENDING',ASR=f"{x['ASR']:.1%}" if x.get('ASR') is not None else 'N/A',role=x['role']) for x in confirmation]
    text=['# Stage 1 当前结果汇总：Uniform可靠，精简选层已有候选优势','',f"数据截点：{snapshot['as_of_UTC']}。这是进行中的可复现快照，不是整轮实验已完成声明。",'',
      '当前最明确的结果：三代理均匀四层是稳定基线；在30图开发集上，每代理三层在InternVL8B达到29/30（Uniform26/30），每代理两层在Gemma E4B达到28/30（Uniform27/30）。这些候选的独立50图/大Target验证需逐项查看，不能把开发选优当成最终提升。','',
      '## 1. 科学设定与数据角色','',
      '每个代理、每个原生视觉层：L=0.5(1−cos(z,μt))+0.5(1+cos(z,μs))。模型内先平均层损失，再求像素梯度；每代理逐图L1归一化，跨代理等权融合，动量符号更新。50步、224×224、L∞16/255、步长1/255、动量1、DI开启、随机起点和逐图固定RNG，种子42。全部模型参数BF16且不量化；48张/类参考图，与攻击图分离。没有语言输出、十类margin或patch loss。','',
      '五方向：金鱼→蝴蝶、菠萝→吉他、笔记本→浓缩咖啡、火山→摇椅、足球→校车。screen10为历史开发数据；explore30与confirm50为独立图片队列。开发验证模型是Qwen9B/InternVL8B/Gemma E4B；正式大模型确认是27B/14B/31B。每个实际目标排除整个同家族代理。Stage3的150图正式集仍封存。','',
      '**指标：定向命中数/N，N为全部预先固定的GT图片。** 另存clean-correct条件指标。无proxy分类成功门槛；本轮没有可用proxy输出条件TASR，不能把两种分母混用。共享图片跨三个目标的观察相关，不能当成3N张独立图片。','',
      '## 2. 论文方法试验完成表','',mdtable(count_rows,['family','method','complete','planned']),'',
      '条件可以被多个方法复用，表中行数不能简单相加当成唯一新攻击次数。ILA是独立的源侧投影/传播诊断；FDA是借鉴条件式联合评估的PP适配，没有原论文辅助分类器；PRAF也是PP梯度对齐适配。AdaEA/CWA/FiT/SEA用于设计和诊断参考，尚无忠实复现结果。','',
      '## 3. 开发10图方法选择结果','',mdtable(methods,['family','method','hits_N','layers']),'',
      'Beam保留的多个候选全部在paper_method_candidates.csv中；这里给出候选中最高命中值，只用于开发描述，存在多重选优偏差。不得据此宣称显著改进。总体上Greedy/Beam未稳定超过Uniform；独立选代理再按单层排名不可靠。弱单层+低CKA对照不能单独产生有效迁移，低相似度不是充分条件。','',
      '## 4. 三代理2/3/4层：30图配对结果','',mdtable(n30,['family','method','hits_N','paired']),'',
      '配对字段gained/lost保留新增与丢失命中；不是将单层成功集合取并集。更少tap不代表前后向计算或耗时等比例减少：一个代理的多层损失仍由一次联合反传求梯度，且视觉塔可能照常执行全部块。实际耗时还受同期并发和放置策略影响。','',
      '## 5. 四层位置对照','',mdtable([dict(family=x['family'],scheme=x['scheme'],hits_N=f"{x['hits']}/{x['N']}" if x['complete'] else 'PENDING') for x in positions.get('rows',[]) if x['split']=='explore'],['family','scheme','hits_N']),'',
      'Uniform为25/50/75/100%：24块模型取[5,11,17,23]，16块模型取[3,7,11,15]，均为零基原生块ID。其余位置方案见four_layer_position_results.csv。没有一个替代四层位置在所有家族上都更好。','',
      '## 6. 50图确认与大Target','',mdtable(c50,['model','method','hits_N','ASR','role']),'',
      '原主方法selected最终等同Uniform，原商业测试共100个唯一clean/adv请求，并不是两个独立方法各测试100次。新精选来自三代理2/3层开发试验；FDA新分支已冻结开发10图选型并新增30/50图与固定三代理控制。确认50沿用既有Uniform已评估的队列，应称后续配对确认，不能称全新未接触测试。','',
      '## 7. 商业验证与新增FDA','',
      f"原GPT-6.1 Sol low：{[(m['method'],m['hits'],m['N']) for m in commercial.get('methods',[])]}。请求模型由订阅Codex CLI指定；服务端具体snapshot未披露，不能冒充直接API或其他GPT版本。",'',
      '新商业对照为Uniform/精选/FDA，同一50图、同一prompt和low设置，复用校验一致的clean/Uniform请求。配置在新请求前冻结，旧Uniform结果已知需披露；不会根据GPT反馈调参，也不会自动扩到Gemini/Claude。新进度见fda_selected_commercial/summary.json，未返回的不是零。','',
      '## 8. 机制测量与解释边界','',mdtable(measurement_files,['file','rows','mtime_UTC']),'',
      '现有测量包括逐层表示移动、类离散度/原型margin、CKA/RSA、梯度cosine、负点积、符号冲突及融合抵消；旧CSV刷新时间单独保留，不假称覆盖所有最新条件。步/层/图像行高度相关，不可当作独立统计样本。V8本轮PCA/t-SNE尚未生成。','',
      '可支持的观察：条件互补性与非单调增益存在；强单组件不保证最佳组合；均匀覆盖深度是可靠启发式。尚不能证明共享语义共识是迁移的唯一原因、ILA/FDA原方法失败、低CKA保证互补、或已经蒸馏出万能视觉头。','',
      '## 9. 校验、失败与未完成项','',f"核验结果：{validation['result']}；逐图重算分子/分母和原文解析，检查{len(pngs)}个PNG哈希及{len(checked_pairs)}对实际像素扰动；BF16审计事件{len(audits)}。详见validation.json。",'',
      '保留历史主机OOM、270条件额度停止及修复记录；已完成PNG没有覆盖。当前上限8 Worker，RAM物理cap与显存实际余量共同准入，大Target优先。原截止时间不重置，未完成项按PARTIAL/QUEUED报告。','',
      '尚需收齐：新精选/FDA 30/50与大模型及商业比较；更新机制汇总与可视化；跨seed42/43/44、COCO/新类别、JPEG/resize、忠实论文baseline属于后续验证，不能算已完成。','',
      '## 10. 文件与复现','',
      'all_conditions.csv：所有条件、角色、层/代理组成、命中/N与证据路径。per_image_evaluations.csv：逐图评分。per_direction.csv：方向细分。attack_receipts.csv：攻击时长、反传数、峰值、PNG路径。paper_method_candidates.csv：Greedy/Beam等全部保留候选。三种method_lock与商业对照锁均单独复制。snapshot.json含输入JSON的SHA256与截点。','',
      '复现汇总：`python scripts/summarize_v8_stage1.py`；该脚本仅CPU读证据，不加载模型、不发商业请求、不接管GPU。文献与适配差异见仓库stage12_joint24h/literature_matrix.md及PROXY_SELECTION_LITERATURE.md。']
    (out/'STAGE1_REPORT.md').write_text('\n'.join(text)+'\n')
    C.atomic(C.ROOT/'stage1_summary/LATEST.json',dict(path=str(out),as_of_UTC=snapshot['as_of_UTC'],validation=validation['result']))
    print(json.dumps(dict(output=str(out),validation=validation,counts=snapshot['counts']),ensure_ascii=False))
    return 0 if validation['result']=='PASS' else 2

if __name__=='__main__':sys.exit(main())
