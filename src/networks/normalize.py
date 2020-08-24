import torch.nn as nn

def get_norm_layer(dims, normtype):
    if normtype == 'batchnorm':
        norm = nn.BatchNorm2d(dims)
    elif normtype == 'instancenorm':
        norm = nn.InstanceNorm2d(dims, affine=True)
    elif normtype == 'layernorm':
        norm = nn.LayerNorm(dims)
    else:
        norm = None
    return norm
