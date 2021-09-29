import torch
import torch.nn as nn
from zenTransformer import zenTransformer


class SiameseModel(nn.Module):
    def __init__(self, subtransformer):
        super().__init__()
        self.subnet = subtransformer        

    def forward(self, src, tgt1, tgt2, src_mask, tgt_mask):
        output1 = self.subnet(src, tgt1, src_mask=src_mask, tgt_mask=tgt_mask)
        output2 = self.subnet(src, tgt2, src_mask=src_mask, tgt_mask=tgt_mask)
        return output1, output2