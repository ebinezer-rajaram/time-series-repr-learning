import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["model"].get("input_dim", None)  # may need to set
        channels = 64
        kernel_size = 3
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.squeeze(-1)  # remove pooled dim
        return x
