import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    1D CNN encoder for time series.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    input_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) → (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x  # (batch, seq_len, hidden_dim)
