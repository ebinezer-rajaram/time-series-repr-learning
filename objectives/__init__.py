from typing import Callable
from .forecasting import mse_loss, mae_loss
from .masking import reconstruction_loss
from .contrastive import nt_xent_loss


def get_loss(name: str, **kwargs) -> Callable:
    """
    Return the appropriate loss function for pretraining.
    Args:
        name: Name of the loss/objective.
            One of: forecasting_mse, forecasting_mae, masking, contrastive
        kwargs: Optional keyword arguments for loss (e.g., temperature for contrastive)
    """
    name = name.lower()

    if name in ("forecasting", "forecasting_mse"):
        return mse_loss
    elif name == "forecasting_mae":
        return mae_loss
    elif name == "masking":
        return lambda y_pred, y_true, mask: reconstruction_loss(y_pred, y_true, mask)
    elif name == "contrastive":
        temperature = kwargs.get("temperature", 0.5)
        return lambda z_i, z_j: nt_xent_loss(z_i, z_j, temperature=temperature)
    else:
        raise ValueError(
            f"Unknown pretraining objective: {name}. "
            "Expected one of: forecasting(_mse), forecasting_mae, masking, contrastive."
        )
