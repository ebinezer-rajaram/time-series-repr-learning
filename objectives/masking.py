import torch
import torch.nn.functional as F

def reconstruction_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss for masked inputs.
    Only computes loss on masked positions.
    Args:
        y_pred: [batch_size, seq_len, ...] — model output
        y_true: [batch_size, seq_len, ...] — original input
        mask:   [batch_size, seq_len] — binary mask (1=masked, 0=visible)
    """
    # mask: [B, T] → [B, T, 1, ...] to broadcast
    while mask.ndim < y_pred.ndim:
        mask = mask.unsqueeze(-1)
    diff = (y_pred - y_true) * mask
    return (diff**2).sum() / mask.sum().clamp_min(1.0)
