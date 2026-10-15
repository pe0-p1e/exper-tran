import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();path=a.output_dir/'audits/frozen_target_generation_smoke.json'
 if not path.is_file():raise SystemExit(1)
 rows=json.loads(path.read_text())
 if len(rows)!=3 or any(x.get('status')!='pass' or len(x.get('outputs',[]))!=4 for x in rows):raise SystemExit(1)
 print('PASS all target generators evaluated only on four frozen smoke PNGs')
if __name__=='__main__':main()
