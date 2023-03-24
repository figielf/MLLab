import torch.nn as nn

from transformers_and_attention.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d, d_model, n_heads, dropout_prob=0.1, is_casual=False, max_len=None, ann_h_dim=4, add_causal_question_attention=False):
        super().__init__()

        self.add_causal_question_attention = add_causal_question_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if add_causal_question_attention:
            self.norm3 = nn.LayerNorm(d_model)
            self.causal_self_mha = MultiHeadAttention(d, d_model, n_heads, is_casual=True, max_len=max_len)
        if is_casual:
            assert max_len is not None and isinstance(max_len, int)
            self.mha = MultiHeadAttention(d, d_model, n_heads, is_casual=True, max_len=max_len)
        else:
            self.mha = MultiHeadAttention(d, d_model, n_heads)
        self.ann = nn.Sequential(
            # two layer ANN used in BERT paper as opposed to single layer ANN used in 'All you need is Attention' paper
            nn.Linear(d_model, ann_h_dim),
            nn.GELU(),
            nn.Linear(ann_h_dim, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, xq, xk, xv, padding_mask=None, output_padding_mask=None):  # xq shape -> (N, T_output, d_model), xk and xv shape -> (N, T_input, d_model), padding_mask shape -> (N, T_input), output_padding_mask shape -> (N, T_output)
        if self.add_causal_question_attention:
            cs_mha_out = self.causal_self_mha(xq, xq, xq, output_padding_mask)  # -> (N, T_output, d_model)
            xq = self.norm3(xq + cs_mha_out)  # -> (N, T_output, d_model)
        mha_out = self.mha(xq, xk, xv, padding_mask)  # -> (N, T_output, d_model)
        x = self.norm1(xq + mha_out)  # -> (N, T_output, d_model)
        ann_out = self.ann(x)  # -> (N, T_output, d_model)
        x = self.norm2(x + ann_out)  # -> (N, T_output, d_model)
        x = self.dropout(x)  # -> (N, T_output, d_model)
        return x
