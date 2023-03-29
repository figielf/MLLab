import torch.nn as nn

from transformers_and_attention.decoder import TextTranslationDecoder
from transformers_and_attention.encoder import EmbeddingEncoder


class TextTranslationTransformer(nn.Module):
    def __init__(self, encoder_vocab_size, encoder_max_len, decoder_vocab_size, decoder_max_len, d, d_model, n_heads, n_layers, dropout_prob):
        super().__init__()

        self.encoder = EmbeddingEncoder(
            vocab_size=encoder_vocab_size,
            max_len=encoder_max_len,
            d=d,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            task_type='many_to_many')
        self.decoder = TextTranslationDecoder(
            vocab_size=decoder_vocab_size,
            max_len=decoder_max_len,
            d=d,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob)

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):  # enc_input shape -> (N, T_input), dec_input shape -> (N, T_output)
        encoder_output = self.encoder(enc_input, enc_mask)  # -> (N, T_input, d_model)
        decoder_output = self.decoder(encoder_output, dec_input, enc_mask, dec_mask)  # -> (N, T_output, n_classes)
        return decoder_output
