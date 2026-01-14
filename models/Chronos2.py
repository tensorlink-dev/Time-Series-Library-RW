import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

try:
    from transformers import BertConfig, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class _Model(nn.Module):
    def __init__(self, configs):
        """
        Chronos-2 is an encoder-only transformer model.
        Initialize with random weights using BertConfig.
        """
        super().__init__()
        # Hardcoded BERT config similar to Chronos-2
        config = BertConfig(
            vocab_size=4096,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
        )
        self.model = BertModel(config)
        self.pred_head = nn.Linear(768, 1)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device

        # Process each channel through encoder
        outputs = []
        for i in range(C):
            channel_data = x_enc[:, :, i]  # [B, L]
            # Quantize to token IDs
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)

            # Get encoder output
            encoder_output = self.model(input_ids=input_ids).last_hidden_state  # [B, L, hidden]

            # Use last hidden states to predict future
            pred = self.pred_head(encoder_output[:, -self.pred_len:, :]).squeeze(-1)  # [B, pred_len]
            outputs.append(pred)

        dec_out = torch.stack(outputs, dim=-1)  # [B, pred_len, C]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None


# Export Model only if transformers is available
Model = _Model if TRANSFORMERS_AVAILABLE else None
