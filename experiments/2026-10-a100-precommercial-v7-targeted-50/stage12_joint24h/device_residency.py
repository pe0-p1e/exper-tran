"""Move only the exact participating vision module; preserve every BF16 parameter."""
import time,types

def visual_module(proxy):
    family=proxy.spec['family'];model=proxy.model
    if family=='qwen':return model.model.visual
    if family in ['internvl','gemma4']:return model.model.vision_tower
    if family in ['clip','siglip2']:return model.vision_model
    if family=='dinov2':return model
    raise ValueError('Unknown vision family '+family)

def install(proxy):
    # Audit was performed by the loader. CPU placement does not change parameter dtype.
    proxy.model.to('cpu')
    proxy._v8_visual=visual_module(proxy)
    def to(self,device):
        start=time.perf_counter();self._v8_visual.to(device)
        self.offload_seconds+=time.perf_counter()-start
        return self
    proxy.to=types.MethodType(to,proxy)
    return proxy

def parameter_gib(module):return sum(p.numel()*p.element_size() for p in module.parameters())/2**30

def may_reside(weight_gib,reservation_gib,free_gib,headroom_gib=8):
    # Reference banks and activations remain inside the admitted reservation, not an arbitrary2GiB weight cap.
    budget=max(0,reservation_gib-3)
    return weight_gib<=budget and free_gib>=weight_gib+headroom_gib
