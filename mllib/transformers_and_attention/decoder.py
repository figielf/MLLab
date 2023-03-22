import torch.nn as nn

from transformers_and_attention.positional_encoding import PositionalEncoding
from transformers_and_attention.transformer_block import TransformerBlock


class SequenceClassificationDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, n_classes, dropout_prob):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        t_blocks = [TransformerBlock(d_k, d_model, n_heads, dropout_prob, is_casual=True, max_len=max_len) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*t_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.final_classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):  # x shape -> (N, T)
        x = self.emb(x)  # -> (N, T, d_model)
        x = self.pos_encoding(x)  # -> (N, T, d_model)
        for t_block in self.transformer_blocks:
            x = t_block(x, mask)  # -> (N, T, d_model)
        x = self.ln(x)  # -> (N, T, n_classes)
        output = self.final_classifier(x)  # -> (N, T, n_classes)
        #output = F.softmax(output, dim=-1)
        return output
