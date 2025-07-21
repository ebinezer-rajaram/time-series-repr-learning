import torch
import torch.nn.functional as F

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss for forecasting.
    Args:
        y_pred: [batch_size, seq_len, ...]
        y_true: [batch_size, seq_len, ...]
    """
    return F.mse_loss(y_pred, y_true, reduction='mean')


def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error loss for forecasting.
    """
    return F.l1_loss(y_pred, y_true, reduction='mean')
