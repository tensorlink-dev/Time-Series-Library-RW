import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import T5Config, T5ForConditionalGeneration


class Model(nn.Module):
    def __init__(self, configs):
        """
        Chronos-Bolt is based on T5 architecture.
        Initialize with random weights using T5Config.
        """
        super().__init__()
        # Hardcoded T5 config similar to Chronos-Bolt-Base
        self.d_model = 512
        config = T5Config(
            vocab_size=4096,
            d_model=self.d_model,
            d_kv=64,
            d_ff=2048,
            num_layers=6,
            num_heads=8,
            dropout_rate=0.1,
            decoder_start_token_id=0,
            pad_token_id=0,
        )
        self.model = T5ForConditionalGeneration(config)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Projection layers for differentiable training path
        self.input_projection = nn.Linear(1, self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Always use differentiable path (encoder forward + projection)
        outputs = []
        for i in range(C):
            channel_data = x_enc[:, :, i].unsqueeze(-1)  # [B, L, 1]
            embedded = self.input_projection(channel_data)  # [B, L, d_model]
            hidden = self.model.encoder(inputs_embeds=embedded).last_hidden_state
            pred_values = self.output_projection(hidden[:, -1, :])  # [B, pred_len]
            outputs.append(pred_values)
        dec_out = torch.stack(outputs, dim=-1)  # [B, pred_len, C]

        # Denormalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
