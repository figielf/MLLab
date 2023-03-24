import torch.nn as nn
import torch.nn.functional as F

from transformers_and_attention.positional_encoding import PositionalEncoding
from transformers_and_attention.transformer_block import TransformerBlock


class EmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d, d_model, n_heads, n_layers, dropout_prob, task_type='many_to_one'):
        assert task_type in ('many_to_one', 'many_to_many')
        super().__init__()

        self.task_type = task_type
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        t_blocks = [TransformerBlock(d, d_model, n_heads, dropout_prob=dropout_prob, ann_h_dim=d_model * 4) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*t_blocks)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None, return_logits=False):  # x shape -> (N, T)
        x = self.emb(x)  # -> (N, T, d_model)
        x = self.pos_encoding(x)  # -> (N, T, d_model)
        for t_block in self.transformer_blocks:
            x = t_block(x, x, x, padding_mask)  # -> (N, T, d_model)

        if self.task_type == 'many_to_one':
            x = x[:, 0, :]  # (N, d_model), only one point is needed as we use bidirectional self attentions here

        output = self.ln(x)  # -> (N, d_model) for 'many_to_one' or  (N, T, d_model) for 'many_to_many'
        if return_logits:
            output = F.softmax(output, dim=-1)
        return output


class TextClassificationEncoder(EmbeddingEncoder):
    def __init__(self, vocab_size, max_len, d, d_model, n_heads, n_layers, n_classes, dropout_prob):
        super().__init__(vocab_size=vocab_size, max_len=max_len, d=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout_prob=dropout_prob, task_type='many_to_one')
        self.final_classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, padding_mask=None, return_logits=False):  # x shape -> (N, T)
        x = super().forward(x, padding_mask=padding_mask, return_logits=return_logits)  # -> x shape -> (N, T)
        output = self.final_classifier(x)  # -> (N, n_classes)
        if return_logits:
            output = F.softmax(output, dim=-1)
        return output
