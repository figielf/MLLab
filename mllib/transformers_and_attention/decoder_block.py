import torch.nn as nn

from transformers_and_attention.transformer_block import TransformerBlock


class SelfCausalDecoderBlock(nn.Module):
    def __init__(self, d, d_model, n_heads, max_len, dropout_prob=0.1, is_casual=True):
        super().__init__()
        self.transformer_block = TransformerBlock(d, d_model, n_heads, dropout_prob=dropout_prob, is_casual=is_casual, max_len=max_len, ann_h_dim=d_model * 4)

    def forward(self, x, padding_mask=None):  # x shape -> (N, T, d_model)
        x = self.transformer_block(x, x, x, padding_mask)  # -> (N, T, d_model)
        return x


class InputOutputDecoderBlock(nn.Module):
    def __init__(self, d, d_model, n_heads, max_len, dropout_prob=0.1, is_casual=True):
        super().__init__()
        self.transformer_block = TransformerBlock(d, d_model, n_heads, dropout_prob=dropout_prob, is_casual=True, max_len=max_len, ann_h_dim=d_model * 4, add_causal_question_attention=True)

    def forward(self, x_input, x_output, input_padding_mask=None, output_padding_mask=None):  # x_input shape -> (N, T_input, d_model), x_output shape -> (N, T_output, d_model), input_padding_mask shape -> (N, T_input), output_padding_mask shape -> (N, T_output)
        x = self.transformer_block(xq=x_output, xk=x_input, xv=x_input, padding_mask=input_padding_mask, output_padding_mask=output_padding_mask)  # -> (N, T, d_model)
        return x
