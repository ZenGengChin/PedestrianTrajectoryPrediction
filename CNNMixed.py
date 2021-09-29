import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch import Tensor
from PTPfunction import generate_mask
import math

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

        
class CNNMixedTransformer(nn.Module):
    def __init__(self, d_coord = 2, d_model:int = 512, 
                 num_encoder_layer:int = 6,
                 num_decoder_layer:int = 6,
                 nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.2
                 ):
        self.d_coord = d_coord
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layer,
            num_decoder_layers = num_decoder_layer,
            dim_feedforward = dim_feedforward,
            dropout = dropout, batch_first = True,
            activation='relu'
        )
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 6, 3), nn.ReLU(),nn.MaxPool2d(), 
            nn.Conv2d(6, 12, 4), nn.ReLU(),nn.MaxPool2d(),
            nn.Conv2d(12, 24, 6), nn.ReLU(),nn.MaxPool2d(),
            nn.Linear(24 * 6 * 6, d_model)
        )
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout)
        self.LinearEmbedding1 = nn.Linear(d_coord, d_model)
        self.LinearEmbedding2 = nn.Linear(d_coord, d_model)
        self.Generator = nn.Linear(d_model, d_coord)

    def forward(self, src: Tensor, tgt: Tensor, src_mask, tgt_mask, image):
        # Image form [N, C, H, W] cnn embedding form [N, dmodel]
        cnn_embedding = self.CNN(image).unsqueeze(1)
        coord_encoding = self.positional_encoding(self.LinearEmbedding1(src))
        coord_decoding = self.positional_encoding(self.LinearEmbedding2(tgt))
        encoder_input = torch.cat((cnn_embedding, coord_encoding), dim=1)
        decoder_input = torch.cat((cnn_embedding, coord_decoding), dim=1)

        out = self.transformer(coord_encoding, coord_decoding, src_mask, tgt_mask)
        return self.Generator(out)
    def encode(self, src: Tensor, src_mask, image):
        cnn_embedding = self.CNN(image).unsqueeze(1)
        coord_encoding = self.positional_encoding(self.LinearEmbedding1(src))
        encoder_input = torch.cat((cnn_embedding, coord_encoding), dim=1)
        return self.transformer.encoder(
            self.positional_encoding(encoder_input,src_mask))

    def decode(self, tgt: Tensor, memory: Tensor, image):
        '''memory is the output by the encoder'''
        cnn_embedding = self.CNN(image).unsqueeze(1)
        coord_decoding = self.positional_encoding(self.LinearEmbedding2(tgt))
        decoder_input = torch.cat((cnn_embedding, coord_decoding), dim=1)
        return self.transformer.decoder(
            self.positional_encoding(decoder_input,memory))
    
    def greedy_decode(self, src, src_mask, max_len, image):
        '''src in form of [N, L, F]'''
        memory = self.encode(src, src_mask, image)
        ys = self.CNN(image).unsqueeze(1)#torch.zeros((src.size(0),1,self.d_input))
        for i in range(max_len):
            tgt_mask = generate_mask(ys.size(1))            
            out = self.decode(ys, memory)
            ys = torch.cat((ys, self.Generator(out)[:,-1:,:]), dim=1)
        return ys[:,1:,:]