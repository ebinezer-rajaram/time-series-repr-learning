import torch.nn as nn
from models import build_model
from models.heads import ForecastingHead, MaskingHead, ContrastiveHead
from objectives import get_loss

def build_full_model(config):
    encoder = build_model(
        config['model']['name'],
        input_dim=config['model']['input_dim'],
        model_dim=config['model'].get('model_dim', 64),
        hidden_dim=config['model'].get('hidden_dim', 64),
        num_layers=config['model'].get('num_layers', 2),
        num_heads=config['model'].get('num_heads', 4),
        dim_feedforward=config['model'].get('dim_feedforward', 128),
        dropout=config['model'].get('dropout', 0.1)
    )

    obj_type = config['objective']['type'].lower()
    if obj_type == "forecasting":
        model = ForecastingHead(encoder, config['model']['model_dim'], config['model']['input_dim'], config['objective']['pred_len'])
    elif obj_type == "masking":
        model = MaskingHead(encoder, config['model']['model_dim'], config['model']['input_dim'])
    elif obj_type == "contrastive":
        model = ContrastiveHead(encoder, config['model']['model_dim'], config['objective'].get('projection_dim', 64))
    else:
        raise ValueError(f"Unsupported objective type: {obj_type}")

    return model, encoder

def build_loss(config):
    return get_loss(config['objective']['type'], **config['objective'])
