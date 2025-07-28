import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Computes the normalized temperature-scaled cross entropy loss (NT-Xent).
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    N = z_i.shape[0]

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    )

    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z_i.device)
    logits = similarity_matrix / temperature

    loss = F.cross_entropy(logits, labels)
    return loss


def embedding_norms(z: torch.Tensor) -> dict:
    """
    Computes mean and std of embedding L2 norms.
    """
    norms = torch.norm(z, dim=1)
    return {
        "embedding_norm_mean": norms.mean().item(),
        "embedding_norm_std": norms.std().item()
    }


def cosine_similarity_stats(z: torch.Tensor, labels: torch.Tensor = None) -> dict:
    """
    Computes mean positive and negative cosine similarity within a batch.
    Assumes labels (if given) define positive pairs.
    """
    z = F.normalize(z, dim=1).cpu().numpy().ravel()
    sim_matrix = cosine_similarity(z)

    if labels is None:
        return {
            "cosine_sim_mean": np.mean(sim_matrix),
            "cosine_sim_std": np.std(sim_matrix)
        }

    labels = labels.cpu().numpy()
    pos_mask = labels[:, None] == labels[None, :]
    neg_mask = ~pos_mask

    pos_sim = sim_matrix[pos_mask]
    neg_sim = sim_matrix[neg_mask]

    return {
        "positive_cosine_sim_mean": np.mean(pos_sim),
        "negative_cosine_sim_mean": np.mean(neg_sim),
        "positive_negative_gap": np.mean(pos_sim) - np.mean(neg_sim)
    }


def batch_entropy(logits: torch.Tensor) -> float:
    """
    Computes entropy of softmax probabilities over a batch.
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean().item()


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes linear Centered Kernel Alignment (CKA) between two matrices.
    X: (n_samples, n_features_1)
    Y: (n_samples, n_features_2)
    """
    def gram_linear(A):
        return A @ A.T

    def center(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    K = center(gram_linear(X))
    L = center(gram_linear(Y))

    hsic = np.trace(K @ L)
    norm_K = np.trace(K @ K)
    norm_L = np.trace(L @ L)

    return hsic / (np.sqrt(norm_K) * np.sqrt(norm_L) + 1e-8)


def embedding_rank(z: torch.Tensor) -> dict:
    """
    Computes the rank and singular value spectrum of embeddings.
    """
    z_np = z.detach().cpu().numpy()
    U, S, Vt = svd(z_np, full_matrices=False)
    explained_variance_ratio = S / np.sum(S)

    return {
        "embedding_rank": np.sum(S > 1e-5),
        "singular_values": S.tolist(),
        "explained_variance_ratio": explained_variance_ratio.tolist()
    }


def compute_all(z: torch.Tensor, z_pos: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor = None) -> dict:
    """
    Computes a set of pretraining metrics for logging.
    """
    metrics = {}
    metrics.update(embedding_norms(z))
    # metrics.update(cosine_similarity_stats(z, labels))

    if z_pos is not None:
        metrics["nt_xent_loss"] = nt_xent_loss(z, z_pos).item()
        
    if logits is not None:
        metrics["batch_entropy"] = batch_entropy(logits)


    metrics.update(embedding_rank(z))
    return metrics
