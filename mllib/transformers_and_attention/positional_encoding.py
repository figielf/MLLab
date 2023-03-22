import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_prob)

        # prepare positional encoding for later use
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(max_len)
        feature_id = torch.arange(0, d_model, 2)
        i2_arg = feature_id * -math.log(10000) / d_model
        feature_id_hash = torch.exp(i2_arg)
        trig_arg = t.unsqueeze(1) * feature_id_hash.unsqueeze(0)
        pe[:, 0::2] = torch.sin(trig_arg)
        pe[:, 1::2] = torch.cos(trig_arg[:, :d_model // 2])
        pe = pe.view(1, max_len, d_model)
        # register pe so that its not considered as module parameter
        self.register_buffer('pe', pe)

    def forward(self, x):  # x shape=(N, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]  # x shape=(N, T, d_model)
        x = self.dropout(x)
        return x
