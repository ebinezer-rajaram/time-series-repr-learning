"""
Visualisations for pretraining embeddings and diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from sklearn.manifold import TSNE
import umap


def plot_tsne(embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
              save_path: Optional[Path] = None, title: str = "t-SNE Embedding") -> plt.Figure:
    """
    Plots 2D t-SNE projection of embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6,6))
    if labels is not None:
        scatter = ax.scatter(z_2d[:,0], z_2d[:,1], c=labels, cmap='tab10', s=10, alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        ax.scatter(z_2d[:,0], z_2d[:,1], s=10, alpha=0.7)

    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_umap(embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
              save_path: Optional[Path] = None, title: str = "UMAP Embedding") -> plt.Figure:
    """
    Plots 2D UMAP projection of embeddings.
    """
    reducer = umap.UMAP(random_state=42)
    z_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6,6))
    if labels is not None:
        scatter = ax.scatter(z_2d[:,0], z_2d[:,1], c=labels, cmap='tab10', s=10, alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        ax.scatter(z_2d[:,0], z_2d[:,1], s=10, alpha=0.7)

    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                            save_path: Optional[Path] = None,
                            title: str = "Pairwise Cosine Similarity") -> plt.Figure:
    """
    Plots heatmap of pairwise similarity matrix.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(similarity_matrix, cmap='coolwarm', ax=ax, cbar=True)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_singular_values(singular_values: np.ndarray,
                         save_path: Optional[Path] = None,
                         title: str = "Singular Value Spectrum") -> plt.Figure:
    """
    Plots singular value spectrum of embedding matrix.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(singular_values, marker='o')
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_attention_map(attention_weights: np.ndarray,
                       save_path: Optional[Path] = None,
                       title: str = "Attention Map") -> plt.Figure:
    """
    Plots heatmap of attention weights.
    Expects shape (seq_len, seq_len) or (heads, seq_len, seq_len).
    """
    fig, ax = plt.subplots(figsize=(6,5))
    if attention_weights.ndim == 3:  # heads x seq x seq
        attention_weights = np.mean(attention_weights, axis=0)
    sns.heatmap(attention_weights, cmap='viridis', ax=ax, cbar=True)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity


class PlotLogger:
    """
    Automatically saves plots and metrics for a specific experiment.
    """
    def __init__(self, output_dir: Path, experiment_name: str = "experiment"):
        self.base_dir = Path(output_dir) / experiment_name / "plots"
        self.experiment_dir = Path(output_dir) / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(self, fig: plt.Figure, name: str):
        """
        Saves a matplotlib Figure under the experiment's plots folder.
        """
        path = self.base_dir / f"{name}.png"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

    def tsne(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None, name: str = "tsne"):
        fig = plot_tsne(embeddings, labels, title=f"{name.upper()}")
        self.save_plot(fig, name)

    def umap(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None, name: str = "umap"):
        fig = plot_umap(embeddings, labels, title=f"{name.upper()}")
        self.save_plot(fig, name)

    def similarity(self, sim_matrix: np.ndarray, name: str = "similarity"):
        fig = plot_similarity_heatmap(sim_matrix, title=f"{name.upper()}")
        self.save_plot(fig, name)

    def singular_values(self, singular_values: np.ndarray, name: str = "singular_values"):
        fig = plot_singular_values(singular_values, title=f"{name.upper()}")
        self.save_plot(fig, name)

    def attention(self, attn_weights: np.ndarray, name: str = "attention"):
        fig = plot_attention_map(attn_weights, title=f"{name.upper()}")
        self.save_plot(fig, name)

    def all(self, embeddings: np.ndarray, labels: np.ndarray, metrics: dict, encoder, step: int, logger=None):
        """
        Save all diagnostic plots and metrics in a clean, reproducible way.
        """

        self.tsne(embeddings, labels)
        self.umap(embeddings, labels)

        sim_matrix = cosine_similarity(embeddings)
        self.similarity(sim_matrix)

        if "singular_values" in metrics:
            self.singular_values(np.array(metrics["singular_values"]))

        if hasattr(encoder, "get_last_attention_map"):
            attn_weights = encoder.get_last_attention_map()
            self.attention(attn_weights)

        print(f"🎨 All plots saved under {self.base_dir}")



