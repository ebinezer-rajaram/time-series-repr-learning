from models.transformer import CustomTransformerEncoder
from models.lstm import LSTMEncoder
from models.cnn import CNNEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn


def get_model(config):
    model_type = config["model"]["type"]

    if model_type == "transformer":
        return CustomTransformerEncoder(config)
    
    elif model_type == "baseline_transformer":
        d_model = config["model"]["d_model"]
        n_heads = config["model"]["n_heads"]
        num_layers = config["model"]["num_layers"]
        dropout = config["model"]["dropout"]

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        return nn.Sequential(
            encoder,
            nn.AdaptiveAvgPool1d(1),  # optional: reduce seq_len
        )
    
    elif model_type == "lstm":
        return LSTMEncoder(config)
    
    elif model_type == "cnn":
        return CNNEncoder(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
