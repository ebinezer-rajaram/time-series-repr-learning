from .lstm import LSTMEncoder
from .cnn import CNNEncoder
from .vae import VAEEncoder
from .transformer import BaselineTransformer

__all__ = [
    "LSTMEncoder",
    "CNNEncoder",
    "VAEEncoder",
    "BaselineTransformer",
    "build_model",
]

_MODEL_REGISTRY = {
    "lstm": LSTMEncoder,
    "cnn": CNNEncoder,
    "vae": VAEEncoder,
    "transformer": BaselineTransformer,
}

def build_model(name: str, **kwargs):
    """
    Build a model by name.
    Example:
        model = build_model("lstm", input_dim=10, hidden_dim=64)
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)
