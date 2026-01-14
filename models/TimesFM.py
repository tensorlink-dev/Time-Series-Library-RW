import torch
from torch import nn

# Try to import transformers, fallback to pure PyTorch if unavailable
try:
    from transformers import GPT2Config, GPT2Model
    HAS_GPT2 = True
except ImportError:
    HAS_GPT2 = False


class FallbackDecoderModel(nn.Module):
    """Fallback decoder model when transformers GPT2 is not available."""
    def __init__(self, n_embd=1024, n_layers=24, n_heads=16):
        super().__init__()
        self.embed = nn.Embedding(4096, n_embd)
        decoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x = self.decoder(x)
        return type('Output', (), {'last_hidden_state': x})()


class Model(nn.Module):
    def __init__(self, configs):
        """
        TimesFM is a decoder-only foundation model for time series.
        Initialize with random weights using GPT2Config.
        """
        super().__init__()
        n_embd = 1024
        if HAS_GPT2:
            # Hardcoded GPT2 config similar to TimesFM-500M
            config = GPT2Config(
                vocab_size=4096,
                n_positions=2048,
                n_embd=n_embd,
                n_layer=24,
                n_head=16,
            )
            self.model = GPT2Model(config)
        else:
            self.model = FallbackDecoderModel(n_embd=n_embd, n_layers=24, n_heads=16)
        self.pred_head = nn.Linear(n_embd, 1)
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

        # Process each channel separately
        outputs = []
        for i in range(C):
            channel_data = x_enc[:, :, i]  # [B, L]
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)

            # Get model output
            hidden = self.model(input_ids=input_ids).last_hidden_state  # [B, L, hidden]
            pred = self.pred_head(hidden[:, -self.pred_len:, :]).squeeze(-1)  # [B, pred_len]
            outputs.append(pred)

        dec_out = torch.stack(outputs, dim=-1)  # [B, pred_len, C]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
