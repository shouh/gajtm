# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.score_function = score_function
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, key, query, mask=None):
        k_len = key.shape[1]
        q_len = query.shape[1]
        if self.score_function == 'dot_product':
            score = torch.bmm(query, key.permute(0, 2, 1))
        elif self.score_function == 'scaled_dot_product':
            qkt = torch.bmm(query, key.permute(0, 2, 1))
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            keyx = torch.unsqueeze(key, dim=1).expand(-1, q_len, -1, -1)
            queryx = torch.unsqueeze(query, dim=2).expand(-1, -1, k_len, -1)
            score = F.tanh(torch.matmul(torch.cat((keyx, queryx), dim=-1), self.weight))
        elif self.score_function == 'bi_linear':
            score = torch.bmm(torch.matmul(query, self.weight), key.permute(0, 2, 1))
        else:
            raise RuntimeError('invalid score_function')
        if mask is not None:
            # 使用tensor的mask_fill方法，将掩码张量每个位置进行比较，如果等于0，用-1e9来代替
            score = score.masked_fill(mask, -1e9)
            # print(score)
        score = F.softmax(score, dim=-1) # 在最后一纬进行softmax操作
        output = torch.bmm(score, key)
        output = self.dropout(output)
        return output, score
