import copy
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from category_data import parse_class_code,classification_prompt,validate_dictionary,validate_nested_dictionaries
from run_v7_attack import objective
from v7_common import fingerprint


def test_new_parser_accepts_all_C_labels_without_relaxing_original_first_line_rule():
    assert parse_class_code('49',50)==dict(label=50,status='ok')
    assert parse_class_code('10',10)['status']=='invalid'
    assert parse_class_code('10',20)['label']==11
    assert parse_class_code('03',50)['status']=='invalid'
    assert parse_class_code('3 extra',50)['status']=='invalid'
    assert parse_class_code('\n3\nexplanation',50)['label']==4
    assert parse_class_code('',50)['status']=='empty'


@pytest.mark.parametrize('C',[12,50])
def test_existing_visual_objective_has_finite_pixel_gradients_beyond_ten_classes(C):
    torch.manual_seed(4)
    class Proxy:
        def __init__(self):
            self.weight=torch.randn(3,16)
            self.bank={l:(torch.randn(C,48,16),torch.randn(C,8,16)) for l in [0,1]}
        def features(self,x):
            tokens=x.flatten(2).transpose(1,2)@self.weight
            z=tokens.mean(1);mask=torch.ones(tokens.shape[:2],dtype=torch.bool)
            return {l:(z,tokens,mask) for l in [0,1]}
    x=torch.rand(2,3,8,8,requires_grad=True)
    loss,_=objective(Proxy(),x,1,C,dict(loss='margin',aggregation='centroid',patch_weight=.35))
    grad=torch.autograd.grad(loss.sum(),x)[0]
    assert loss.shape==(2,) and torch.isfinite(loss).all()
    assert torch.isfinite(grad).all() and (grad.flatten(1).abs().sum(1)>0).all()


def dictionary():
    def row(cid,kind,i):
        iid=f'{cid}/{kind}{i}'
        return dict(image_id=iid,class_id=cid,path=iid+'.png',sha256=fingerprint(iid))
    classes=[dict(id=f'new{i}',label=i+1,name=f'new category{i}',references=[row(f'new{i}','ref',j) for j in range(48)]) for i in range(12)]
    return dict(class_count=12,classes=classes,transitions=[dict(id=f'G{i}',source_label=i*2+1,target_label=i*2+2,images=[row(f'new{i*2}','attack',j) for j in range(30)]) for i in range(5)],excluded_original_class_ids=['original'])


def test_dictionary_rejects_reference_leakage_and_source_label_errors(tmp_path):
    data=dictionary();validate_dictionary(data,tmp_path,check_files=False)
    overlap=copy.deepcopy(data);overlap['transitions'][0]['images'][0]=overlap['classes'][0]['references'][0]
    with pytest.raises(ValueError,match='overlap'):validate_dictionary(overlap,tmp_path,check_files=False)
    wrong=copy.deepcopy(data);wrong['transitions'][0]['images'][0]['class_id']='new11'
    with pytest.raises(ValueError,match='ground-truth'):validate_dictionary(wrong,tmp_path,check_files=False)
    old=copy.deepcopy(data);old['excluded_original_class_ids']=['new1']
    with pytest.raises(ValueError,match='Original'):validate_dictionary(old,tmp_path,check_files=False)


def test_category_prompt_lists_new_dictionary_not_original_categories():
    prompt=classification_prompt(dictionary()['classes'])
    assert '11 new category11' in prompt and '0 to 11' in prompt
    assert 'goldfish' not in prompt


def test_dictionary_scaling_cannot_switch_images_or_reference_banks():
    small=dictionary();large=copy.deepcopy(small)
    large['classes']+= [dict(id='extra',label=13,name='extra class',references=[])]
    assert len(validate_nested_dictionaries([large,small]))==2
    large['transitions'][0]['images'][0]['image_id']='a different image'
    with pytest.raises(ValueError,match='changed images'):validate_nested_dictionaries([small,large])


def test_generalized_prompt_matches_original_protocol_for_original_ten_classes():
    import yaml
    from common import classification_prompt as original_prompt
    from v7_common import V6
    config=yaml.safe_load((V6/'config/runner_template.yaml').read_text())
    assert classification_prompt(config['classes'])==original_prompt(config)


def test_frozen_attack_checks_actual_clean_pixels_and_png_budget(tmp_path):
    import hashlib
    import json
    import numpy as np
    from PIL import Image
    from run_category_generalization import validate_frozen_attacks
    from v7_common import sha
    clean=np.full((224,224,3),100,dtype=np.uint8)
    adv=clean+16
    Image.fromarray(clean).save(tmp_path/'clean.png')
    Image.fromarray(adv).save(tmp_path/'adv.png')
    images=[dict(image_id=str(i),pixel_sha256=hashlib.sha256(clean.tobytes()).hexdigest()) for i in range(30)]
    data=dict(classes=[{},{}],transitions=[dict(id='G01',source_label=1,target_label=2,images=images)])
    state=dict(status='attack_complete',manifest_sha256='manifest',steps=50,seed=42,restarts=1,target_access=False,models=['proxy'],class_count=2,config=dict(patch_weight=.35,input_diversity=True),source_label=1,target_label=2,image_ids=[r['image_id'] for r in images],clean_paths=['clean.png']*30,adv_paths=['adv.png']*30,png_hashes={p:sha(tmp_path/p) for p in ['clean.png','adv.png']})
    dest=tmp_path/'G01/batch_000';dest.mkdir(parents=True)
    (dest/'state.json').write_text(json.dumps(state))
    validate_frozen_attacks(tmp_path,data,'manifest',['proxy'])
    data['transitions'][0]['images'][0]['pixel_sha256']='wrong cohort pixels'
    with pytest.raises(RuntimeError,match='clean pixels'):validate_frozen_attacks(tmp_path,data,'manifest',['proxy'])
    data['transitions'][0]['images'][0]['pixel_sha256']=hashlib.sha256(clean.tobytes()).hexdigest()
    Image.fromarray(clean+17).save(tmp_path/'adv.png')
    state['png_hashes']['adv.png']=sha(tmp_path/'adv.png')
    (dest/'state.json').write_text(json.dumps(state))
    with pytest.raises(RuntimeError,match='perturbation'):validate_frozen_attacks(tmp_path,data,'manifest',['proxy'])
