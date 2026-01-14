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
    def __init__(self, n_embd=512, n_layers=8, n_heads=8):
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
        Moirai-2 is a decoder-only transformer for time series.
        Initialize with random weights using GPT2Config.
        """
        super().__init__()
        n_embd = 512
        if HAS_GPT2:
            # Hardcoded GPT2 config similar to Moirai-2.0-R-small
            config = GPT2Config(
                vocab_size=4096,
                n_positions=1024,
                n_embd=n_embd,
                n_layer=8,
                n_head=8,
            )
            self.model = GPT2Model(config)
        else:
            self.model = FallbackDecoderModel(n_embd=n_embd, n_layers=8, n_heads=8)
        self.pred_head = nn.Linear(n_embd, 1)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        outputs = []
        for i in range(C):
            channel_data = x_enc[:, :, i]
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)
            hidden = self.model(input_ids=input_ids).last_hidden_state
            pred = self.pred_head(hidden[:, -self.pred_len:, :]).squeeze(-1)
            outputs.append(pred)

        dec_out = torch.stack(outputs, dim=-1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
