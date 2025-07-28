import torch.nn as nn

class ForecastingHead(nn.Module):
    """
    Wraps a general encoder and projects its output for forecasting.
    Outputs only the last `pred_len` steps, mapped back to input_dim.
    """
    def __init__(self, encoder, model_dim, input_dim, pred_len):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(model_dim, input_dim)
        self.pred_len = pred_len

    def forward(self, x):
        h = self.encoder(x)               # [B, seq_len, model_dim]
        h = self.proj(h)                 # [B, seq_len, input_dim]
        return h[:, -self.pred_len:, :]  # [B, pred_len, input_dim]


class MaskingHead(nn.Module):
    """
    Wraps a general encoder and projects output back to input_dim.
    Typically used with a mask during loss computation.
    """
    def __init__(self, encoder, model_dim, input_dim):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        h = self.encoder(x)              # [B, seq_len, model_dim]
        h = self.proj(h)                # [B, seq_len, input_dim]
        return h


class ContrastiveHead(nn.Module):
    """
    Optional: wraps encoder and projects pooled embedding to latent space.
    """
    def __init__(self, encoder, model_dim, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(model_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)             # [B, seq_len, model_dim]
        h = h.mean(dim=1)              # global pooling: [B, model_dim]
        z = self.proj(h)               # [B, projection_dim]
        return z
