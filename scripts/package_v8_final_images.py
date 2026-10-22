#!/usr/bin/env python3
"""Publish exact frozen final150 PNG pairs; no inference or image modification."""
import json,sys,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'experiments/2026-10-a100-precommercial-v7-targeted-50/stage12_joint24h'))
import common as C
from paper_closeout.plan import OUT
DEST=C.EXP/'reports/v8_paper_closeout/final_images'

def main():
    C.check_lock();DEST.mkdir(parents=True,exist_ok=True)
    lock=C.read(OUT/'final_method_lock.json');cases=C.cohort('final150')['images'];index=[]
    for family,methods in lock['families'].items():
        name={'qwen':'qwen27b_and_all_gpt','internvl':'internvl14b','gemma4':'gemma31b'}[family]
        target=DEST/f'final150_{name}_clean_adv.zip';temp=target.with_suffix('.tmp')
        manifest=dict(family=family,N=150,methods={},cohort_sha256=C.sha(C.ROOT/'manifests/final150.json'),clean_shared=True,
            commercial_images=family=='qwen',selection_used_GPT50=True,final150_used_for_selection=False,images=[])
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for method,v in methods.items():
                c=v['condition'];manifest['methods'][method]=c
                for row in cases:
                    receipt=C.image_receipt(c,row['image_id'])
                    if not C.valid_receipt(receipt,c):raise RuntimeError('Invalid frozen PNG receipt: '+str(receipt))
                    state=C.read(receipt);relative=Path(row['image_id']).with_suffix('.png')
                    clean_name='clean/'+relative.as_posix();adv_name='adversarial/'+method+'/'+relative.as_posix()
                    if method==next(iter(methods)):
                        z.write(state['clean'],clean_name)
                    else:
                        first=C.read(C.image_receipt(next(iter(methods.values()))['condition'],row['image_id']))
                        if C.sha(first['clean'])!=C.sha(state['clean']):raise RuntimeError('Clean PNG identity differs across paired methods')
                    z.write(state['adv'],adv_name)
                    manifest['images'].append(dict(method=method,image_id=row['image_id'],transition=row['transition'],source=row['source'],target=row['target'],
                        clean=clean_name,adversarial=adv_name,clean_sha256=C.sha(state['clean']),adversarial_sha256=C.sha(state['adv']),linf=state['linf_png']))
            z.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
            z.writestr('README.md','# Frozen final150 paired images\n\n150 shared clean PNGs and450 adversarial PNGs: uniform, selected and FDA, each150. Pair using manifest.json. These are original evaluated PNG bytes, not resized or reconstructed.\n\nQwen27B/GPT uses its family-excluded proxy pool; InternVL14B and Gemma31B use their own legal pools. All methods use50steps, epsilon16/255, pure1:1 Pull/Push, DI and equal L1-normalized gradients. Exact layers/configurations and PNG hashes are in manifest.json. No final150 feedback was used for tuning. FDA denotes an adaptation, not auxiliary-classifier FDA reproduction.\n')
            prompt=next((OUT/'commercial150/packs/gpt-6.1-sol/uniform/blinded').glob('*/prompt.txt'),None)
            if prompt:z.write(prompt,'commercial_evaluation_prompt.txt')
        with zipfile.ZipFile(temp) as z:
            if z.testzip():raise RuntimeError('Corrupt image archive')
            pngs=[n for n in z.namelist() if n.endswith('.png')]
            if len(pngs)!=600:raise RuntimeError('Incomplete clean/adv archive')
        temp.replace(target);index.append(dict(family=family,file=target.name,sha256=C.sha(target),bytes=target.stat().st_size,
            clean_PNGs=150,adversarial_PNGs=450,pairs=450,methods=list(methods)))
    C.atomic(DEST/'index.json',dict(archives=index,all_final150_families=True,PNG_total_including_shared_clean_per_archive=1800,
        unique_adversarial_images=1350,clean_unique_across_families=150,commercial_subset='final150_qwen27b_and_all_gpt_clean_adv.zip'))
    print(json.dumps(index,indent=2))
if __name__=='__main__':main()
