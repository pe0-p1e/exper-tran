"""Explicit raw arithmetic mean control, with no gradient normalization."""

def raw_equal_mean(gradients,*,mode='equal',weights=None,**kwargs):
    if gradients.ndim!=5 or mode!='equal' or weights is not None:
        raise ValueError('Raw control only supports equal unweighted [K,B,C,H,W]')
    return gradients.float().mean(0)
