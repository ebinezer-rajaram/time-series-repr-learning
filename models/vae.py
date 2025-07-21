import torch.nn as nn

class VAEEncoder(nn.Module):
    """
    Variational Autoencoder encoder for time series.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        **kwargs
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # outputs both μ and logσ²
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        b, t, d = x.shape
        x = x.view(b * t, -1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return (
            mu.view(b, t, -1),
            logvar.view(b, t, -1),
        )
