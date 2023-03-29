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
        self.max_len = max_len

        if is_casual:
            assert self.max_len is not None and isinstance(max_len, int)
            cm = torch.tril(torch.ones(max_len, max_len))  # -> (max_len, max_len)
            self.register_buffer('causal_mask', cm)
        else:
            self.register_buffer('causal_mask', None)

    def forward(self, xq, xk, xv, padding_mask=None):  # xq shape=(N, T_input, d_model), xk and xv shape=(N, T_output, d_model), attention_mask shape=(N, T)
        Q = self.query(xq)  # -> (N, T_output, h * d)
        K = self.key(xk)  # -> (N, T_input, h * d)
        V = self.value(xv)  # -> (N, T_input, h * d)

        N = Q.shape[0]
        T_output = Q.shape[1]
        T_input = K.shape[1]
        if self.max_len is not None:
            assert T_output <= self.max_len
            assert T_input < self.max_len

        # reshape (N, T, h * d) -> (N, h, T, d)
        Q = Q.view(N, T_output, self.h, self.d).transpose(1, 2)  # -> (N, h, T_output, d)
        K = K.view(N, T_input, self.h, self.d).transpose(1, 2)  # -> (N, h, T_input, d)
        V = V.view(N, T_input, self.h, self.d).transpose(1, 2)  # -> (N, h, T_input, d)

        # compute attention weights
        attention_score = Q @ K.transpose(-2, -1)  # -> (N, h, T_output, T_input), same as torch.matmul(Q, K.mT)
        attention_score = attention_score / math.sqrt(self.d)  # -> (N, h, T_output, T_input)

        # apply masks
        if padding_mask is not None:
            attention_score = attention_score.masked_fill(padding_mask[:, None, None, :] == 0, float('-inf'))  # -> (N, h, T_output, T_input)
        if self.causal_mask is not None:
            attention_score = attention_score.masked_fill(self.causal_mask[None, None, :T_output, :T_input] == 0, float('-inf'))  # -> (N, h, T, T)
        attention_weights = F.softmax(attention_score, dim=-1)  # -> (N, h, T, T)

        # compute attention
        attention = attention_weights @ V  # -> (N, h, T_output, d), same as torch.matmul(attention_weights, V)

        # concatenate attention heads
        attention = attention.transpose(1, 2)  # -> (N, T_output, h, d)
        attention = attention.contiguous().view(N, T_output, self.h * self.d)  # -> (N, T_output, h * d), same as attention.contiguous().view(N, T_output, self.h * self.d) or attention.reshape(N, T_output, -1)

        # compute multi-head attention
        multihead = self.final_dense(attention)  # -> (N, T_output, d_model)
        return multihead
