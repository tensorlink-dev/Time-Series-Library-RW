import torch
from torch import nn

# Try to import transformers, fallback to pure PyTorch if unavailable
try:
    from transformers import T5Config, T5ForConditionalGeneration
    HAS_T5 = True
except ImportError:
    HAS_T5 = False


class FallbackChronosModel(nn.Module):
    """Fallback model when transformers T5 is not available."""
    def __init__(self, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(4096, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, 4096)

    def generate(self, input_ids, max_new_tokens, **kwargs):
        x = self.embed(input_ids)
        x = self.encoder(x)
        logits = self.output_proj(x)
        # Simple greedy generation
        generated = [input_ids]
        last_hidden = x[:, -1:, :]
        for _ in range(max_new_tokens):
            logits = self.output_proj(last_hidden)
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            last_hidden = self.embed(next_token)
        return torch.cat(generated, dim=1)


class Model(nn.Module):
    def __init__(self, configs):
        """
        Chronos-Bolt is based on T5 architecture.
        Initialize with random weights using T5Config.
        """
        super().__init__()
        if HAS_T5:
            # Hardcoded T5 config similar to Chronos-Bolt-Base
            config = T5Config(
                vocab_size=4096,
                d_model=512,
                d_kv=64,
                d_ff=2048,
                num_layers=6,
                num_heads=8,
                dropout_rate=0.1,
            )
            self.model = T5ForConditionalGeneration(config)
        else:
            self.model = FallbackChronosModel(d_model=512, n_layers=6, n_heads=8)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        device = x_enc.device

        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Process each channel and generate predictions
        outputs = []
        for i in range(C):
            # Prepare input for T5 (quantize to token ids)
            channel_data = x_enc[:, :, i]  # [B, L]
            # Simple quantization to vocab range
            input_ids = ((channel_data + 3) * 100).long().clamp(0, 4095)  # [B, L]

            # Generate future tokens
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.pred_len,
                do_sample=False,
            )
            # Dequantize back to values
            pred_tokens = generated[:, -self.pred_len:]  # [B, pred_len]
            pred_values = (pred_tokens.float() / 100) - 3
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
