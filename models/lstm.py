import torch.nn as nn

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for time series.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        output, (hn, cn) = self.lstm(x)
        return output  # (batch, seq_len, hidden_dim*2 if bidirectional)
