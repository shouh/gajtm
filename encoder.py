# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, num_layer = 1):
        super(Encoder, self).__init__()
        self.pe = PositionEmbedding(input_size, input_size)
        self.layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dropout = 0.4)
        self.tr = nn.TransformerEncoder(self.layer, num_layers=1)

    def forward(self, inputs):
        inputs = self.pe(inputs)
        out = self.tr(inputs.permute(1, 0, 2)).permute(1, 0, 2)
        return out


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.uniform_(self.pe.weight, -0.1, 0.1)

    def forward(self, x):
        b, l, d = x.size()
        seq_len = torch.arange(l).to(x.device)
        return x + self.pe(seq_len).unsqueeze(0)


# performance poor
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
