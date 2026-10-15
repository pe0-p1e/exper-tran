#!/usr/bin/env python3
"""Verify each frozen target generator only after the smoke adversarial PNG exists."""
import argparse,gc,json,os,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'experiments/2026-09-a100-final-v5/src'),str(ROOT/'src')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=REPO/'outputs/a100_precommercial_v6_multi_repr_recovery');a=p.parse_args();out=a.output_dir.resolve()
 state_path=out/'multiple_proxy/attacks/smoke_qwen2b_clip_dino/T01.json'
 if not state_path.is_file():raise FileNotFoundError('Frozen two-image smoke attack must finish first')
 state=json.loads(state_path.read_text());paths=[]
 for b in state['batches']:
  paths.extend([out/x for pair in zip(b['clean_paths'],b['adv_paths'],strict=True) for x in pair])
 if len(paths)!=4 or any(not x.is_file() for x in paths):raise RuntimeError('Expected two frozen clean/adversarial PNG pairs')
 lock=json.loads((ROOT/'model_revisions.json').read_text());matrix=yaml.safe_load((ROOT/'experiment_matrix.yaml').read_text())['multiple_proxy'];targets=sorted({r['target'] for b in ('block_1','block_2') for r in matrix[b]})
 from common import classification_prompt
 from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
 from primary_ml_cka.models.backends.target_transformers_generation import load_target_for_generation
 from primary_ml_cka.models.backends.transformers_backend import load_processor
 from primary_ml_cka.models.common.loading import local_snapshot
 from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
 from strict_model_audit import audit_loaded_model
 prompt=classification_prompt(yaml.safe_load((ROOT/'config/runner_template.yaml').read_text()));result=[]
 for target in targets:
  spec=lock[target];MODEL_REVISIONS[spec['repo_id']]=spec['revision'];model=processor=generator=None
  try:
   snapshot=local_snapshot(Path(os.environ['HF_HOME']),spec['repo_id'],spec['revision']);processor=load_processor(snapshot);model=load_target_for_generation(snapshot,torch.device('cuda'),precision='bf16')
   audit_loaded_model(model,model_name=target,revision=spec['revision'],role='target_evaluator',repo_id=spec['repo_id'],output_dir=out)
   generator=TransformersTargetGenerator(model,processor)
   with torch.inference_mode():outputs=generator.generate_labels(paths,prompt)
   if len(outputs)!=4:raise RuntimeError(f'{target}: returned {len(outputs)} outputs for four frozen PNGs')
   result.append({'target':target,'revision':spec['revision'],'status':'pass','outputs':[{'raw_output':x.raw_output,'parsed_label':x.parsed_label,'parser_status':x.parser_status} for x in outputs]})
   print(f'[frozen-target-smoke] PASS {target} generated {len(outputs)} frozen PNG outputs',flush=True)
  except Exception as exc:
   result.append({'target':target,'revision':spec['revision'],'status':'error','error':f'{type(exc).__name__}: {exc}'});raise
  finally:
   if generator is not None:del generator
   if model is not None:del model
   if processor is not None:del processor
   gc.collect();torch.cuda.empty_cache()
 path=out/'audits/frozen_target_generation_smoke.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
