"""Uniform four-layer V6 cosine Pull/Push; no margin or patch loss."""
import torch
from representation_objectives import original_pull_push_loss, class_scores


def objective(proxy,x,source,target,config):
    if config['loss']!='pull_push' or config.get('patch_weight',0)!=0 or config.get('projector_weight',0)!=0:
        raise ValueError('Pure Pull/Push does not include margin, patch or projector losses')
    pull,push=config['pull_weight'],config['push_weight']
    if pull<0 or push<0 or abs(pull+push-1)>1e-8:raise ValueError('Weights must sum to one')
    features=proxy.features(x);losses=[];margins=[]
    for layer,(z,_,_) in features.items():
        bank=proxy.bank[layer][0]
        losses.append(original_pull_push_loss(z,bank,source,target,pull_weight=pull,push_weight=push,reduction='none'))
        # Descriptive all-class margin only. It is never part of the objective.
        with torch.no_grad():
            scores=class_scores(z,bank);other=scores.clone();other[:,target-1]=-torch.inf
            margins.append(scores[:,target-1]-other.max(1).values)
    if len(losses)!=len(config['layers']):raise RuntimeError('Missing actual visual layer tap')
    return torch.stack(losses).mean(0),torch.stack(margins).mean(0)
