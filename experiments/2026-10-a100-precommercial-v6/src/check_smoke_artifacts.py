import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();o=a.output_dir
 path=o/'multiple_proxy/attacks/smoke_qwen2b_clip_dino/T01.json'
 if not path.is_file():raise SystemExit(1)
 s=json.loads(path.read_text())
 if s.get('status')!='attack_complete' or len(s.get('image_ids',[]))!=2 or len(s.get('batches',[]))!=1:raise SystemExit(1)
 b=s['batches'][0]
 if b.get('steps')!=2 or b.get('gradient_fusion')!='mean_of_per_image_l1_normalized_proxy_gradients':raise SystemExit(1)
 if len(b.get('adv_paths',[]))!=2 or not all((o/x).is_file() for x in b['adv_paths']):raise SystemExit(1)
 print('PASS frozen heterogeneous multi-proxy smoke artifacts')
if __name__=='__main__':main()
