# objectives/contrastive.py

import torch
import torch.nn.functional as F

def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
    Encourages z_i and z_j to be close while pushing others apart.
    
    Args:
        z_i: [batch_size, dim] — embedding of first view
        z_j: [batch_size, dim] — embedding of second view
        temperature: scaling factor
    """
    batch_size = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)  # [2B, 2B]
    logits = similarity_matrix / temperature

    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    logits = logits.masked_fill(mask, float('-inf'))

    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ], dim=0)

    loss = F.cross_entropy(logits, positives)
    return loss
