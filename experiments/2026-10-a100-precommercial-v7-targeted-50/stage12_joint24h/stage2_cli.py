"""50-image adapter around the already-tested blinded subscription CLI runner."""
from common import *
import argparse,shutil,subprocess

def prepare_packs(count=50):
 lock=read(ROOT/'method_lock.json')
 if not lock:raise RuntimeError('Stage2 requires immutable method lock')
 from common import make
 family=lock['stage2_primary_family'];cases=cohort('confirm')['images'];assert len(cases)==50
 from stage1_methods.early_confirm import ordered_cases,frozen
 order_cases=ordered_cases(cases)[:count]
 if count not in [30,50]:raise ValueError('Only30/50 confirmation packets')
 packet_root=ROOT/('stage2/packs30' if count==30 else 'stage2/packs')
 source=next((DATA/'commercial_pilot10/blinded').glob('*/prompt.txt'));prompt=source.read_text()
 packs=[]
 for method in lock['stage2_methods']:
  layers=lock['families'][family]['primary' if method=='selected' else 'uniform'];c=make(family,layers,split='confirm',label=method)
  if not frozen(c,count):raise RuntimeError('Missing frozen confirmation attacks for '+method)
  pack=packet_root/method;private=[];reqs=[];order=[]
  for r in order_cases:
   d=read(image_receipt(c,r['image_id']));ids={}
   for kind,key in [('clean','clean'),('adversarial','adv')]:
    path=Path(d[key]);rid='r_'+digest(['V8',kind,sha(path),digest(prompt),'gpt-6.1-sol','low'])[:24];ids[kind]=rid
    dest=pack/'blinded'/rid;dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest/'image.png');(dest/'prompt.txt').write_text(prompt)
    reqs.append(dict(request_id=rid,image_sha256=sha(path),prompt_sha256=sha(dest/'prompt.txt')));order.append(rid)
   private.append(dict(image_id=r['image_id'],transition=r['transition'],source_label=r['source'],target_label=r['target'],requests=ids))
  freeze(pack/'private/manifest.json',dict(cases=private,requests=reqs));freeze(pack/'request_order.json',order);packs.append(pack)
 freeze(ROOT/('stage2/pack_lock30.json' if count==30 else 'stage2/pack_lock.json'),dict(method_lock_sha256=sha(ROOT/'method_lock.json'),packs={p.name:sha(p/'private/manifest.json') for p in packs},prospective_unique_requests=len({rid for p in packs for rid in read(p/'request_order.json')}),quota_bound_seconds_at180=27000,feedback_for_selection=False))
 return packs

def run_legacy(pack,limit):
 import run_codex_cli_pilot as old
 original_add=old.argparse.ArgumentParser.add_argument
 def add(parser,*args,**kwargs):
  if '--images' in args:kwargs['choices']=[3,10,30,50,150]
  return original_add(parser,*args,**kwargs)
 old.argparse.ArgumentParser.add_argument=add
 original_check=old.checked_order
 def checked(base,count):
  if count not in [30,50]:return original_check(base,count)
  d=read(base/'private/manifest.json');order=read(base/'request_order.json');declared={r['request_id']:r for r in d['requests']}
  ids=[rid for c in d['cases'] for rid in c['requests'].values()]
  if len(d['cases'])!=count or len(set(c['image_id'] for c in d['cases']))!=count or len(order)!=2*count or len(set(order))!=2*count or set(order)!=set(ids):raise ValueError('Invalid isolated50-image packet')
  for rid in order:
   p=base/'blinded'/rid
   if sha(p/'image.png')!=declared[rid]['image_sha256'] or sha(p/'prompt.txt')!=declared[rid]['prompt_sha256']:raise ValueError('Changed Stage2 packet')
  return order
 old.checked_order=checked
 count=len(read(pack/'private/manifest.json')['cases'])
 sys.argv=[sys.argv[0],'--images',str(count),'--model','gpt-6.1-sol','--pack-dir',str(pack),'--result-dir',str(ROOT/'stage2/responses'),'--limit',str(limit)]
 old.main()
 # Legacy snapshot is copied per method; shared clean request IDs are deliberately reusable.
 shutil.copyfile(ROOT/'stage2/responses/pilot_summary.json',ROOT/'stage2'/(pack.name+f'_N{count}_limit{limit}.json'))

def summarize(count=50):
 methods=[]
 for p in (ROOT/('stage2/packs30' if count==30 else 'stage2/packs')).glob('*'):
  d=read(p/'private/manifest.json');pairs=[]
  if not d:continue
  for c in d['cases']:
   clean=read(ROOT/'stage2/responses'/c['requests']['clean']/'result.json');adv=read(ROOT/'stage2/responses'/c['requests']['adversarial']/'result.json')
   if not clean or not adv:continue
   pairs.append(dict(image_id=c['image_id'],transition=c['transition'],target_hit=adv['parsed_label']==c['target_label'],clean_correct=clean['parsed_label']==c['source_label'],adv_label=adv['parsed_label'],parser_status=adv['parser_status']))
  methods.append(dict(method=p.name,N=count,completed_pairs=len(pairs),hits=sum(x['target_hit'] for x in pairs),ASR=sum(x['target_hit'] for x in pairs)/count if len(pairs)==count else None,
   directions=[dict(transition=t,hits=sum(x['target_hit'] for x in pairs if x['transition']==t),N=count//5,completed=sum(x['transition']==t for x in pairs)) for t in TIDS],images=pairs))
 records=list((ROOT/'stage2/responses').glob('r_*/result.json'))
 complete=len(methods)==2 and all(m['completed_pairs']==count for m in methods)
 atomic(ROOT/('stage2/summary30.json' if count==30 else 'stage2/summary.json'),dict(complete=complete,status='COMPLETE' if complete else 'PARTIAL',methods=methods,unique_completed_requests=len(records),planned_unique_requests=read(ROOT/('stage2/pack_lock30.json' if count==30 else 'stage2/pack_lock.json'),{}).get('prospective_unique_requests',150),
  model='gpt-6.1-sol',reasoning_effort='low',route='existing subscription Codex CLI',actual_server_snapshot='undisclosed unless present in raw traces',method_selected_before_GPT=True))
 return complete

def main():
 p=argparse.ArgumentParser();p.add_argument('--pack');p.add_argument('--limit',type=int,choices=[20,60,100]);p.add_argument('--prepare',action='store_true');a=p.parse_args();check_lock()
 if a.prepare:prepare_packs()
 else:
  try:run_legacy(Path(a.pack),a.limit)
  finally:summarize(count=len(read(Path(a.pack)/'private/manifest.json')['cases']))
if __name__=='__main__':main()
