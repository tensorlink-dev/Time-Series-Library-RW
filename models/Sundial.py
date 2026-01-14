import torch
from torch import nn

# Try to import transformers, fallback to pure PyTorch if unavailable
try:
    from transformers import GPT2Config, GPT2LMHeadModel
    HAS_GPT2 = True
except ImportError:
    HAS_GPT2 = False


class FallbackGenerativeModel(nn.Module):
    """Fallback generative model when transformers GPT2 is not available."""
    def __init__(self, n_embd=512, n_layers=12, n_heads=8, vocab_size=4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        decoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x = self.decoder(x)
        logits = self.lm_head(x)
        return type('Output', (), {'logits': logits})()

    def generate(self, input_ids, max_new_tokens, **kwargs):
        generated = input_ids
        for _ in range(max_new_tokens):
            x = self.embed(generated)
            x = self.decoder(x)
            logits = self.lm_head(x[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class Model(nn.Module):
    def __init__(self, configs):
        """
        Sundial is a flow-based time series generative model.
        Initialize with random weights using GPT2 architecture.
        """
        super().__init__()
        n_embd = 512
        if HAS_GPT2:
            # Use GPT2 config sized similar to Sundial-128M
            config = GPT2Config(
                vocab_size=4096,
                n_positions=1024,
                n_embd=n_embd,
                n_layer=12,
                n_head=8,
            )
            self.model = GPT2LMHeadModel(config)
        else:
            self.model = FallbackGenerativeModel(n_embd=n_embd, n_layers=12, n_heads=8)
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
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.pred_len,
                do_sample=False,
                pad_token_id=0,
            )
            pred_tokens = generated[:, -self.pred_len:]
            pred_values = (pred_tokens.float() / 100) - 3
            outputs.append(pred_values)

        dec_out = torch.stack(outputs, dim=-1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
