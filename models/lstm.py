import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["model"].get("input_dim", None)  # you may need to infer this
        hidden_size = config["model"].get("hidden_size", 64)
        num_layers = config["model"].get("num_layers", 2)
        dropout = config["model"]["dropout"]

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        # x: [B, T, F]
        output, _ = self.lstm(x)
        emb = output.mean(dim=1)  # [B, hidden_size]
            
        return emb
        
