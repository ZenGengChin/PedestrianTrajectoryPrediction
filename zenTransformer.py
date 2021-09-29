import torch 
from torch import Tensor, nn 
from torch.nn import Transformer
from matplotlib import pyplot
import pandas as pd
import numpy as np
import math

from torch.nn.modules import dropout
''' Torch Version : 1.9.1+cpu '''

assert torch.__version__ == '1.9.1+cpu'

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        '''X(tokenembedding) in the form of [N,L,F]'''
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

class zenTransformer(nn.Module):
    def __init__(self, d_input = 4, d_model:int = 512, 
                 num_encoder_layer:int = 6,
                 num_decoder_layer:int = 6,
                 nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.2
                 ):
        super().__init__()
        self.d_input = d_input
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layer,
            num_decoder_layers = num_decoder_layer,
            dim_feedforward = dim_feedforward,
            dropout = dropout, batch_first = True,
            activation='relu'
        )
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)
        self.LinearEmbedding1 = nn.Linear(d_input, d_model)
        self.LinearEmbedding2 = nn.Linear(d_input, d_model)
        self.Generator = nn.Linear(d_model, d_input)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask, tgt_mask):
        src_encoding = self.positional_encoding(self.LinearEmbedding1(src))
        tgt_encoding = self.positional_encoding(self.LinearEmbedding2(tgt))
        out = self.transformer(src_encoding, tgt_encoding, src_mask, tgt_mask)
        return self.Generator(out)
    def encode(self, src: Tensor, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.LinearEmbedding1(src)),src_mask)

    def decode(self, tgt: Tensor, memory: Tensor):
        '''memory is the output by the encoder'''
        return self.transformer.decoder(self.positional_encoding(
            self.LinearEmbedding2(tgt)), memory)
    def greedy_decode(self, src, src_mask, max_len):
        '''src in form of [N, L, F]'''
        memory = self.encode(src, src_mask)
        ys = src[:,-1:,:]#torch.zeros((src.size(0),1,self.d_input))
        for i in range(max_len):
            tgt_mask = generate_mask(ys.size(1))            
            out = self.decode(ys, memory)
            ys = torch.cat((ys, self.Generator(out)[:,-1:,:]), dim=1)
        return ys[:,1:,:]

def generate_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz)) == 1)).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

