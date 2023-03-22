import torch.nn as nn

from transformers_and_attention.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d, d_model, n_heads, dropout_prob=0.1, is_casual=False, max_len=None, ann_h_dim_multiplier=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if is_casual:
            assert max_len is not None and isinstance(max_len, int)
            self.mha = MultiHeadAttention(d, d_model, n_heads, is_casual=True, max_len=max_len)
        else:
            self.mha = MultiHeadAttention(d, d_model, n_heads)
        self.ann = nn.Sequential(
            # ANN used in BERT paper as opposed to single layer ANN used in 'All you need is Attention' paper
            nn.Linear(d_model, d_model * ann_h_dim_multiplier),
            nn.GELU(),
            nn.Linear(d_model * ann_h_dim_multiplier, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, attention_mask=None):  # x shape -> (N, T, d_model)
        mha_out = self.mha(x, x, x, attention_mask)  # -> (N, T, d_model)
        x = self.norm1(x + mha_out)  # -> (N, T, d_model)
        ann_out = self.ann(x)  # -> (N, T, d_model)
        x = self.norm2(x + ann_out)  # -> (N, T, d_model)
        x = self.dropout(x)  # -> (N, T, d_model)
        return x
