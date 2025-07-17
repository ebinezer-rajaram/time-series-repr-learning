import torch.nn as nn

class CustomTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["model"]["d_model"]
        n_heads = config["model"]["n_heads"]
        num_layers = config["model"]["num_layers"]
        dropout = config["model"]["dropout"]
        # TODO: positional encoding, layers, etc.
        self.encoder = nn.Identity()  # placeholder

    def forward(self, x):
        # x: [B, T, F]
        return self.encoder(x)
