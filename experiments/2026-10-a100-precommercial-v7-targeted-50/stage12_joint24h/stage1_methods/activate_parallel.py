"""Compile exact owner integration; activation transaction remains explicit/audited."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as C

def patch_source(source):
    if 'from stage1_methods.parallel_owner import' in source:return source
    source=source.replace('from resources import telemetry,estimate,admit,failure_kind\n',
      'from resources import telemetry,estimate,admit,failure_kind\nfrom stage1_methods.parallel_owner import research_pending,stage2_progress,stop_stage2\n')
    source=source.replace("  if STOP:\n   s['state']='STOPPING'", "  if STOP:\n   stop_stage2()\n   s['state']='STOPPING'")
    source=source.replace("  if time.time()>deadline-3600:\n   for j", "  if time.time()>deadline-3600:\n   stop_stage2()\n   for j")
    needle="   # Prioritize evaluation to release adaptive search prerequisites.\n"
    replacement="""   paper,paper_ready=research_pending(phase)
   paper_background=[(c,'evaluate' if attack_done(c) else 'attack') for c in paper if not eval_done(c)]
   background+=paper_background
   s['paper_research']=dict(pending=len(paper_background),complete=paper_ready,blocked=read(ROOT/'stage1_methods/blocked.json'))
"""+needle
    if source.count(needle)!=1:raise RuntimeError('Changed background API')
    source=source.replace(needle,replacement)
    source=source.replace("pending=fair_pending(pending,background,s['active'],legacy_ids)","pending=fair_pending(pending,background,s['active'],legacy_ids|{c['id'] for c in paper})")
    old="""   if phase=='STAGE1_COMPLETE' and background:phase='PARALLEL_PAIR_COMPLETION'
   if phase=='STAGE1_COMPLETE':
    if s['active']:time.sleep(3);continue
    ok=run_stage2(s,deadline);s['state']='COMPLETE' if ok and not s['blocked'] else 'PARTIAL_OR_BLOCKED';state_write(s);break
"""
    new="""   if phase=='STAGE1_COMPLETE':
    commercial=stage2_progress(deadline);s['commercial_state']=commercial
    if commercial!='RUNNING' and not pending and not s['active']:
     s['state']='COMPLETE' if commercial=='COMPLETE' and paper_ready and not s['blocked'] else 'PARTIAL_OR_BLOCKED';state_write(s);break
    phase='STAGE2_AND_STAGE1_RESEARCH'
    state_write(s)
"""
    if source.count(old)!=1:raise RuntimeError('Changed Stage2 API')
    source=source.replace(old,new)
    source=source.replace("  if not available and not s['active']:\n   s['state']='BLOCKED_REQUIRES_INTERVENTION'", "  if not available and not s['active'] and s.get('commercial_state')=='RUNNING':\n   time.sleep(3);continue\n  if not available and not s['active']:\n   s['state']='BLOCKED_REQUIRES_INTERVENTION'")
    compile(source,'parallel_owner_scheduler','exec')
    return source
if __name__=='__main__':
    import difflib
    src=(C.HERE/'scheduler.py').read_text();new=patch_source(src)
    print(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile='scheduler.py',tofile='scheduler.py')))
