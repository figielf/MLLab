import torch.nn as nn

from transformers_and_attention.positional_encoding import PositionalEncoding
from transformers_and_attention.transformer_block import TransformerBlock


class TextClassificationEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, n_classes, dropout_prob):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        t_blocks = [TransformerBlock(d_k, d_model, n_heads, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*t_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.final_classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):  # x shape -> (N, T)
        x = self.emb(x)  # -> (N, T, d_model)
        x = self.pos_encoding(x)  # -> (N, T, d_model)
        for t_block in self.transformer_blocks:
            x = t_block(x, mask)  # -> (N, T, d_model)

        x = x[:, 0, :]  # only one point is needed as we use bidirectional self attentions here
        x = self.ln(x)  # -> (N, n_classes)
        output = self.final_classifier(x)  # -> (N, n_classes)
        #output = F.softmax(output)
        return output


class SequenceProjectionEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, n_classes, dropout_prob):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        t_blocks = [TransformerBlock(d_k, d_model, n_heads, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*t_blocks)
        self.ln = nn.LayerNorm(d_model)
        #self.final_classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):  # x shape -> (N, T)
        x = self.emb(x)  # -> (N, T, d_model)
        x = self.pos_encoding(x)  # -> (N, T, d_model)
        for t_block in self.transformer_blocks:
            x = t_block(x, mask)  # -> (N, T, d_model)

        #x = x[:, 0, :]  # only one point is needed as we use bidirectional self attentions here
        output = self.ln(x)  # -> (N, n_classes)
        #output = self.final_classifier(x)  # -> (N, n_classes)
        return output
