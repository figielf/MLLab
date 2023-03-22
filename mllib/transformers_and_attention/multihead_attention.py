import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d, d_model, n_heads, is_casual=False, max_len=None):
        super().__init__()

        self.d = d  # all question (dQ), key (dK) and value (dV) vector dimensions are assumed to be the same dimension d
        #self.d_model = d_model  # dimension of input x model size
        self.h = n_heads
        self.query = nn.Linear(d_model, self.h * self.d)
        self.key = nn.Linear(d_model, self.h * self.d)
        self.value = nn.Linear(d_model, self.h * self.d)
        self.final_dense = nn.Linear(self.h * self.d, d_model)

        if is_casual:
            assert max_len is not None and isinstance(max_len, int)
            cm = torch.tril(torch.ones(max_len, max_len))  # -> (max_len, max_len)
            self.register_buffer('causal_mask', cm)
        else:
            self.register_buffer('causal_mask', None)

    def forward(self, xq, xk, xv, attention_mask=None):  # xq, xk, xv are of same shape=(N, T, d_model), T is seq length, attention_mask shape=(N, T)
        Q = self.query(xq)  # -> (N, T, h * d)
        K = self.key(xk)  # -> (N, T, h * d)
        V = self.value(xv)  # -> (N, T, h * d)

        N = Q.shape[0]
        T = Q.shape[1]

        # reshape (N, T, h * d) -> (N, h, T, d)
        Q = Q.view(N, T, self.h, self.d).transpose(1, 2)  # -> (N, h, T, d)
        K = K.view(N, T, self.h, self.d).transpose(1, 2)  # -> (N, h, T, d)
        V = V.view(N, T, self.h, self.d).transpose(1, 2)  # -> (N, h, T, d)

        # compute attention weights
        attention_score = Q @ K.transpose(-2, -1)  # -> (N, h, T, T), same as torch.matmul(Q, K.mT)
        attention_score = attention_score / math.sqrt(self.d)  # -> (N, h, T, T)
        # apply masks
        if attention_mask is not None:
            attention_score = attention_score.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))  # -> (N, h, T, T)
        if self.causal_mask is not None:
            attention_score = attention_score.masked_fill(self.causal_mask[None, None, :T, :T] == 0, float('-inf'))  # -> (N, h, T, T)
        attention_weights = F.softmax(attention_score, dim=-1)  # -> (N, h, T, T)

        # compute attention
        attention = attention_weights @ V  # -> (N, h, T, d), same as torch.matmul(attention_weights, V)

        # concatenate attention heads
        attention = attention.transpose(1, 2)  # -> (N, T, h, d)
        attention = attention.contiguous().view(N, T, -1)  # -> (N, T, h * d), same as attention.contiguous().view(N, T, self.h * self.d) or attention.reshape(N, T, -1)

        # compute multi-head attention
        multihead = self.final_dense(attention)  # -> (N, T, d_model)
        return multihead
