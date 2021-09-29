import torch 
from torch import Tensor
from torch import nn
import numpy as np

def truncSeq(data):
    result = []
    for ins in data:
        if [0 for i in range(0,6)] not in ins.tolist():
            result.append(ins.tolist())
    return np.array(result)

def generate_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def blind_mask(sz: int) -> Tensor:
    return torch.ones(sz, sz) * float('-inf')

def ADE(source, target):
    # Input size is [batch:N, seq:C, dim:M]
    return torch.mean(torch.sqrt(torch.sum((source - target)**2,dim=2)))