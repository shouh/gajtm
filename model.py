import sys
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import warnings
import numpy as np
import math
from attentions import *
from torch.nn import init
from layers import MultiHeadedAttention, ScaledDotProductAttention
from blitz.modules import BayesianGRU, BayesianConv1d, BayesianLinear, BayesianLSTM
from encoder import Encoder
warnings.filterwarnings('ignore')
from glu import GLU

class PMTLM(nn.Module):
    def __init__(self, dropout, learning_rate, margin, hidden_size, word_emb, rel_emb, rel_dict):
        super(PMTLM, self).__init__()
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(th.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False

        self.rel_embedding = nn.Embedding(rel_emb.shape[0], rel_emb.shape[1])
        self.rel_embedding.weight = nn.Parameter(th.from_numpy(rel_emb).float())
        self.rel_embedding.weight.requires_grad = False  # fix the embedding matrix

        self.rel_dict = rel_dict
        self.dropout = True  # dropout 0.35
        self.dropout_rate = dropout
        self.embedding_dim = word_emb.shape[1]  # 300
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate  # 0.001
        self.margin = margin  # 0.5
        self.hidden_er = 300  # 150

        if self.dropout:
            self.rnn_dropout = nn.Dropout(p=self.dropout_rate)

        # Multi-Head Attention
        self.mlhatt = MultiHeadedAttention(8, self.embedding_dim * 2, dropout=self.dropout_rate)

        self.first_gru = Encoder(self.embedding_dim)
        self.second_gru = Encoder(self.embedding_dim)
        self.first_cnn = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=1)

        self.second_cnn = nn.Conv1d(self.embedding_dim * 2, self.embedding_dim, 3, padding=1)

        self.dgconv3 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=3, padding=1, dilation=1)
        self.dgconv4 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=3, padding=2, dilation=2)
        self.dgconv5 = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=3, padding=4, dilation=4)

        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 12, self.hidden_size * 6),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 6, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size * 1, 1)
        )
        """
         Note:
        For SimpleQuestions, when set self.hidden_er to 300, we can obtain the SOTA result
        For WebQuestions, when set self.hidden_er to 150, we can obtain the SOTA result
        """
        self.hidden2tag = nn.Sequential(
            nn.Linear(self.embedding_dim,  150),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(150, 4)
        )
        self.update_layer = nn.Conv1d(600, 300, 3, padding=1)
        self.gate = nn.Conv1d(600, 300, 3, padding=1)
        self.glu = GLU(2, 300, 3, self.dropout_rate, 3000)

        # MultiWay Attentions
        self.sdpatt = Attention(600, score_function='scaled_dot_product', dropout = self.dropout_rate)

        # MLPAttention
        self.mlpatt =  Attention(600, score_function='mlp', dropout = self.dropout_rate)

        self.proj1 = nn.Linear(10 * self.hidden_size, self.hidden_size * 5)
        self.proj2 = nn.Linear(self.hidden_size * 5, self.hidden_size)

        self.proj4 = nn.Linear(5 * self.hidden_size, self.hidden_size * 2)
        self.proj5 = nn.Linear(self.hidden_size * 2, self.hidden_size  * 2)

        self.weight = nn.Parameter(torch.Tensor(self.hidden_size * 2))

    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return th.cat([p1, p2], 1)

    def gated_self_attn(self,  key, query):
        k_len = key.shape[1]
        q_len = query.shape[1]
        keyx = torch.unsqueeze(key, dim=1).expand(-1, q_len, -1, -1)
        queryx = torch.unsqueeze(query, dim=2).expand(-1, -1, k_len, -1)
        score = F.tanh(torch.matmul(torch.cat((keyx, queryx), dim=-1), self.weight))
        score = F.softmax(score, dim=-1) # 在最后一纬进行softmax操作
        output = torch.bmm(score, key)
        # output = self.dropout(output)

        # inputs = output + query
        inputs = th.cat([query, output], dim=2)
        f_t = th.tanh(self.update_layer(inputs.permute(0, 2, 1))) #
        g_t = th.sigmoid(self.gate(inputs.permute(0, 2, 1)))
        updated_output = g_t * f_t + (1 - g_t) * query.permute(0, 2, 1)
        eninput = query.permute(0, 2, 1) + g_t * query.permute(0, 2, 1)
        return updated_output.permute(0, 2, 1), eninput.permute(0, 2, 1)
        # return inputs, inputs

    def forward(self, question, word_relation, rel_relation):
        q_embed = self.word_embedding(question)
        w_embed = self.word_embedding(word_relation)
        """
        KG embedding
        """
        r_embed = self.rel_embedding(rel_relation)

        if self.dropout:
            q_embed = self.rnn_dropout(q_embed)
            w_embed = self.rnn_dropout(w_embed)
            r_embed = self.rnn_dropout(r_embed)

        """
        Bayesian GRU1
        """
        # print(q_embed.size())
        q_encoded = self.first_gru(q_embed)
        w_encoded = self.second_gru(w_embed)
        # r_encoded, _ = self.first_gru(r_embed)

        q_cnn = self.first_cnn(q_embed.permute(0, 2, 1))
        w_cnn = self.first_cnn(w_embed.permute(0, 2, 1))
        # r_cnn = self.first_cnn(r_embed.permute(0, 2, 1))
        q_cnn = q_cnn.permute(0, 2, 1)
        w_cnn = w_cnn.permute(0, 2, 1)
        # r_cnn = r_cnn.permute(0, 2, 1)
        """
        多头自注意力机制
        """

        q_output = th.cat([q_cnn, q_encoded], dim=2)
        w_output = th.cat([w_cnn, w_encoded], dim=2)
        sq_output, sq_weight = self.sdpatt(w_output, q_output)
        mq_output, mq_weight = self.mlpatt(w_output, q_output)

        conc = th.cat([sq_output, mq_output, sq_output - mq_output, sq_output + mq_output, sq_output * mq_output], dim=2)



        proj43 = self.proj2(F.relu(self.proj1(conc)))
        proj2 = self.glu(conc)


        proj1, entiin = self.gated_self_attn(r_embed, proj2.permute(0, 2, 1))
        proj2s, intitn = self.gated_self_attn(r_embed, proj43)

        einput = th.cat([entiin, intitn], dim=2)
        einputatt = self.mlhatt(einput, einput, einput)
        nerinput = einputatt + einput
        eninput = self.second_cnn(nerinput.permute(0, 2, 1))
        e_scores = self.hidden2tag(eninput.permute(0, 2, 1))

        redinput = th.cat([proj1, proj2s, proj1 - proj2s, proj1 + proj2s, proj1 * proj2s], dim=2)
        redinputs = self.proj5(F.relu(self.proj4(redinput)))

        rnn2 = self.dgconv3(redinputs.permute(0, 2, 1))
        rnn3 = self.dgconv4(redinputs.permute(0, 2, 1))
        rnn4 = self.dgconv5(redinputs.permute(0, 2, 1))

        rep = th.cat([rnn2.permute(0, 2, 1), rnn3.permute(0, 2, 1), rnn4.permute(0, 2, 1)], dim=-1)
        """
        分别将w_q_r_output进行最大池化和平均池化
        """
        rnn4 = self.apply_multiple(rep)
        """
        经过MLP层
        """
        score = self.fc(rnn4)
        """
        经过sigmoid激活函数
        """
        score = F.sigmoid(score)
        return e_scores, score, proj1


    def loss_function(self, logits, target, masks, device, num_class=4):
        criterion = nn.CrossEntropyLoss(reduction='none')
        logits = logits.view(-1, num_class)
        target = target.view(-1)
        masks = masks.view(-1)
        cross_entropy = criterion(logits, target)
        loss = cross_entropy * masks
        loss = loss.sum() / (masks.sum() + 1e-12)  # 加上 1e-12 防止被除数为 0
        loss = loss.to(device)
        return loss

if __name__ == '__main__':
    pass
