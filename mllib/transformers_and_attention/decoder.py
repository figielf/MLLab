import torch.nn as nn
import torch.nn.functional as F

from transformers_and_attention.positional_encoding import PositionalEncoding
from transformers_and_attention.transformer_block import TransformerBlock


class SequenceEmbeddingDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, d, d_model, n_heads, n_layers, dropout_prob):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        t_blocks = [self._build_transformer_block(d, d_model, n_heads, max_len, dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*t_blocks)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None, return_logits=False):  # x shape -> (N, T), input_padding_mask shape -> (N, T)
        x = self.emb(x)  # -> (N, T, d_model)
        x = self.pos_encoding(x)  # -> (N, T, d_model)
        for t_block in self.transformer_blocks:
            x = self._call_transformer_block(t_block, x, padding_mask)  # -> (N, T, d_model)
        output = self.ln(x)  # -> (N, T, d_model)
        if return_logits:
            output = F.softmax(output, dim=-1)  # -> (N, T_output, d_model)
        return output

    def _build_transformer_block(self, d, d_model, n_heads, max_len, dropout_prob):
        pass

    def _call_transformer_block(self, transformer_block, x, padding_mask):
        pass


class TextGenerationDecoder(SequenceEmbeddingDecoder):
    def __init__(self, vocab_size, max_len, d, d_model, n_heads, n_layers, dropout_prob):
        super().__init__(vocab_size=vocab_size, max_len=max_len, d=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout_prob=dropout_prob)
        self.final_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, padding_mask=None, return_logits=False):  # x shape -> (N, T)
        x = super().forward(x, padding_mask=padding_mask)  # ->   # -> (N, T, n_classes)
        output = self.final_classifier(x)  # -> (N, T, n_classes)
        if return_logits:
            output = F.softmax(output, dim=-1)
        return output

    def _build_transformer_block(self, d, d_model, n_heads, max_len, dropout_prob):
        return TransformerBlock(d, d_model, n_heads, dropout_prob=dropout_prob, is_casual=True, max_len=max_len, ann_h_dim=d_model * 4)

    def _call_transformer_block(self, t_block, x, padding_mask):
        return t_block(x, x, x, padding_mask)
#

class TextTranslationDecoder(SequenceEmbeddingDecoder):
    def __init__(self, vocab_size, max_len, d, d_model, n_heads, n_layers, dropout_prob):
        super().__init__(vocab_size=vocab_size, max_len=max_len, d=d, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout_prob=dropout_prob)
        self.final_classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x_input, x_output, input_padding_mask=None, output_padding_mask=None, return_logits=False):  # x shape -> (N, T)
        x = super().forward(x_output, mask=output_padding_mask, return_logits=return_logits)  # ->   # -> (N, T, n_classes)
        output = self.final_classifier(x)  # -> (N, T, n_classes)
        if return_logits:
            output = F.softmax(output, dim=-1)
        return output

    def _build_transformer_block(self, d, d_model, n_heads, max_len, dropout_prob):
        return TransformerBlock(d, d_model, n_heads, dropout_prob=dropout_prob, is_casual=False, max_len=max_len, ann_h_dim=d_model * 4, add_causal_self_attention=True)

    def _call_transformer_block(self, transformer_block, x_input, x_output, input_padding_mask, output_padding_mask):
        return transformer_block(x_output, x_input, x_input, input_padding_mask, output_padding_mask)
