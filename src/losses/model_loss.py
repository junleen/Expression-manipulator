import torch
import torch.nn as nn

def get_cycle_loss():
    return nn.L1Loss().cuda()

def get_cond_loss():
    return nn.MSELoss().cuda()


