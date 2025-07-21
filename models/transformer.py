import torch.nn as nn

class BaselineTransformer(nn.Module):
    """
    Transformer Encoder using PyTorch's nn.TransformerEncoder.
    """
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x  # (batch, seq_len, model_dim)
