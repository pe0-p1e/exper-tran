#!/usr/bin/env python3
"""Complete scoped scientific numeric inventory, deduplicate existing publication, chunk gaps."""
import collections,fcntl,gzip,io,json,re,subprocess,sys,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from paper_closeout.plan import OUT
DEST=C.EXP/'reports/v8_paper_closeout/intermediate_supplement'
EXT={'.json','.jsonl','.csv','.npz','.npy'}
SKIP={'responses','blinded','private','logs','auth','credentials','.codex','canonical_images','canonical_final150','deferred_paper_scaffold','commercial_pilot3','commercial_pilot10','commercial_test150'}
SECRET=re.compile(r'(^|_)(api_key|access_token|refresh_token|auth_token|hf_token|password|authorization|client_secret|secret)(_|$)',re.I)
def sensitive(value):
 if isinstance(value,dict):return any((SECRET.search(k) and v not in [None,'',False,[],{}]) or sensitive(v) for k,v in value.items())
 if isinstance(value,list):return any(sensitive(v) for v in value)
 return False

def main():
 C.check_lock();DEST.mkdir(parents=True,exist_ok=True)
 known={}
 for p in (C.EXP/'reports').rglob('*'):
  if p.is_file() and 'intermediate_supplement' not in p.parts and p.suffix in EXT:
   known[C.sha(p)]=str(p.relative_to(C.REPO))
 for p in (C.EXP/'reports/v8_paper_closeout/numerical_receipts').glob('*.jsonl.gz'):
  with gzip.open(p,'rt') as f:
   for i,line in enumerate(f):
    v=json.loads(line);known[v['sha256']]=str(p.relative_to(C.REPO))+f'#record={i}'
 scopes=[('V8',C.ROOT),('V7_exploratory_and_ablation',C.DATA)]
 inventory=[];missing=[];excluded=[]
 for label,root in scopes:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p.is_symlink() or p.suffix not in EXT:continue
   rel=p.relative_to(root)
   if any(part in SKIP for part in rel.parts) or rel.parts[0]=='preflight' or p.name in ['stdout.jsonl','timeout_stdout.jsonl','auth.json','models_cache.json','intermediate_publication_coverage.json']:continue
   if p.suffix in ['.json','.jsonl']:
    text=p.read_text(errors='strict')
    try:values=[json.loads(text)] if p.suffix=='.json' else [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:excluded.append(dict(path=label+'/'+str(rel),reason='malformed/in-progress numeric record'));continue
    if any(sensitive(v) for v in values):excluded.append(dict(path=label+'/'+str(rel),reason='credential-like key; retained local only'));continue
   h=C.sha(p);arc=label+'/'+rel.as_posix();entry=dict(path=arc,sha256=h,bytes=p.stat().st_size)
   if h in known:entry['existing_publication']=known[h]
   else:entry['supplement_member']=arc;missing.append((p,arc))
   inventory.append(entry)
 archive=Path('/home/ubuntu/experiment-checkpoints/v8_numeric_intermediates_supplement.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
 with archive.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=1) as gz:
   with tarfile.open(fileobj=gz,mode='w|') as tar:
    for p,arc in missing:
     info=tar.gettarinfo(str(p),arcname=arc);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.pax_headers={}
     with p.open('rb') as src:tar.addfile(info,src)
 chunks=[]
 with archive.open('rb') as f:
  for i in range(10000):
   data=f.read(20*2**20)
   if not data:break
   p=DEST/f'numeric_intermediates.tar.gz.part{i:03d}';p.write_bytes(data);chunks.append(dict(file=p.name,bytes=len(data),sha256=C.sha(p)))
 manifest=DEST/'file_inventory.json.gz'
 with manifest.open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write((json.dumps(dict(files=inventory,excluded=excluded),separators=(',',':'))+'\n').encode())
 coverage=dict(scopes=[dict(label=label,root=str(root)) for label,root in scopes],scientific_extensions=sorted(EXT),
  indexed_files=len(inventory),already_published_files=sum('existing_publication' in v for v in inventory),supplement_files=len(missing),
  excluded_sensitive_or_malformed=excluded,excluded_categories=['PNG corpora (separate final-image packages)','weights/torch checkpoints','raw commercial sessions/authentication','preflight process/environment snapshots','plain text runtime logs'],
  supplement_bytes=archive.stat().st_size,archive_sha256=C.sha(archive),chunks=chunks,inventory_sha256=C.sha(manifest),
  note='Covers all existing selected numeric file formats in V8 and V7 exploratory/ablation roots. Missing/unexecuted experiments do not have fabricated data. Existing public equivalents are referenced by SHA rather than duplicated.')
 C.atomic(DEST/'coverage.json',coverage);C.atomic(OUT/'reports/intermediate_publication_coverage.json',coverage)
 (DEST/'README.md').write_text('# Complete numerical intermediate supplement\n\nIncludes prior exploratory and ablation measurements that were not already published. file_inventory.json.gz lists every scoped numeric file, its SHA256 and either an existing publication location or a member in this supplement. All source formats .json/.jsonl/.csv/.npy/.npz are inventoried, subject to the explicit privacy exclusions in coverage.json.\n\nReconstruct: `cat numeric_intermediates.tar.gz.part* > numeric_intermediates.tar.gz`, verify archive_sha256 from coverage.json, then `tar -xzf numeric_intermediates.tar.gz`. Existing published equivalents are not duplicated. Raw commercial traces/authentication and non-final image corpora stay local. The final images are separately packaged under ../final_images/. Failed and partial experiments keep their original status.\n')
 print(json.dumps({k:v for k,v in coverage.items() if k not in ['chunks','excluded_sensitive_or_malformed']},indent=2));print('excluded',excluded[:10])
if __name__=='__main__':main()
