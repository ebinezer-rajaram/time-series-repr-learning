import torch
from typing import Tuple, Literal


def apply_random_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random binary mask to x.
    Args:
        x: [B, T, D]
        mask_ratio: fraction of elements to mask
    Returns:
        x_masked: [B, T, D]
        mask: binary mask [B, T, D] where 1 = masked
    """
    mask = (torch.rand_like(x) < mask_ratio).float()
    x_masked = x * (1.0 - mask)
    return x_masked, mask


def split_forecasting(
    x: torch.Tensor,
    pred_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split x into past & future for forecasting.
    Args:
        x: [B, T, D]
        pred_len: number of future steps
    Returns:
        x_input: [B, T-pred_len, D]
        y_target: [B, pred_len, D]
    """
    x_input = x[:, :-pred_len, :]
    y_target = x[:, -pred_len:, :]
    return x_input, y_target


def augment_sequence(
    x: torch.Tensor,
    jitter_std: float = 0.02,
    scaling_std: float = 0.1
) -> torch.Tensor:
    """
    Apply random jitter + scaling augmentation.
    Args:
        x: [B, T, D]
    Returns:
        x_aug: [B, T, D]
    """
    noise = torch.randn_like(x) * jitter_std
    scale = (torch.randn(x.size(0), 1, x.size(2), device=x.device) * scaling_std + 1.0)
    x_aug = (x + noise) * scale
    return x_aug


def prepare_batch(
    x: torch.Tensor,
    objective: Literal["masking", "forecasting", "contrastive"],
    **kwargs
) -> Tuple:
    """
    Dispatcher: prepares batch based on objective.
    Args:
        x: [B, T, D]
        objective: one of 'masking', 'forecasting', 'contrastive'
        kwargs: additional params (e.g., mask_ratio, pred_len)
    Returns:
        Tuple of inputs & targets as needed for the loss
    """
    if objective == "masking":
        x_masked, mask = apply_random_mask(x, **kwargs)
        return x_masked, x, mask
    elif objective == "forecasting":
        x_input, y_target = split_forecasting(x, **kwargs)
        return x_input, y_target
    elif objective == "contrastive":
        x1 = augment_sequence(x, **kwargs)
        x2 = augment_sequence(x, **kwargs)
        return x1, x2
    else:
        raise ValueError(f"Unknown objective: {objective}")
