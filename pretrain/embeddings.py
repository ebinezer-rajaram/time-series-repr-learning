import torch
import numpy as np
from data.dataloader import get_dataloader
from tqdm import tqdm

def reduce_labels_for_plotting(labels: np.ndarray, task: str) -> np.ndarray:
    """
    Reduces raw labels to shape [N] for visualization or evaluation.
    """
    if labels.ndim == 1:
        return labels

    task = task.lower()

    if task == "forecasting":
        return labels[:, 0, 0]  # [N, T, D] → [N]

    elif task == "classification":
        return labels[:, 0] if labels.ndim == 2 else labels  # [N, T] → [N]

    elif task == "masking":
        return labels[:, 0, 0]  # same as forecasting

    elif task == "anomaly_detection":
        return labels.max(axis=1)  # any anomaly in sequence

    elif task == "contrastive":
        return labels[:, 0] if labels.ndim > 1 else labels

    else:
        raise ValueError(f"Unknown task type for label reduction: {task}")


def extract_and_save_embeddings(model, config, device, experiment_dir):
    model.eval()
    loader = get_dataloader(config, split="val")

    embeddings, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting embeddings", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            embeddings.append(outputs.detach().cpu())
            labels.append(targets)
            
    print("Embeddings Before:", torch.cat(embeddings).shape)
    print("Labels Before:", torch.cat(labels).shape)



    embeddings = torch.cat(embeddings, dim=0).numpy()  # [N, T, D]
    embeddings = embeddings.mean(axis=1)               # [N, D]
    
    labels = torch.cat(labels, dim=0).numpy()
    task = config["objective"]["type"]
    labels = reduce_labels_for_plotting(labels, task)
    
    assert embeddings.shape[0] == labels.shape[0], \
        f"Mismatch: {embeddings.shape[0]} embeddings vs {labels.shape[0]} labels"
    
    path = experiment_dir / "embeddings.npz"
    np.savez_compressed(path, embeddings=embeddings, labels=labels)
    print(f"📦 Saved embeddings to {path}")
    print("Embeddings After:", embeddings.shape)
    print("Labels:", labels.shape)

    return embeddings, labels

# def extract_and_save_embeddings(model, config, device, experiment_dir):
#     model.eval()
#     loader = get_dataloader(config, split="val")

#     embeddings, labels = [], []

#     with torch.no_grad():
#         for inputs, targets in tqdm(loader, desc="Extracting embeddings", leave=False):
#             inputs = inputs.to(device)
#             outputs = model(inputs)  # [B, T, D] or [B, D] depending on head
#             embeddings.append(outputs.detach().cpu())  # always [B, T, D] in your setup
#             labels.append(targets.cpu())

#     embeddings = torch.cat(embeddings, dim=0).numpy()  # shape [N, T, D]
#     embeddings = embeddings.mean(axis=1)               # pool across T → [N, D]

#     labels = torch.cat(labels, dim=0).numpy()          # [N, T] or [N, 1] or [N]
#     if labels.ndim > 1:
#         labels = labels[:, 0]                          # 🔥 pick first timestep's label
#     labels = labels.squeeze()                          # [N]

#     assert embeddings.shape[0] == labels.shape[0], \
#         f"Mismatch: {embeddings.shape[0]} embeddings vs {labels.shape[0]} labels"

#     path = experiment_dir / "embeddings.npz"
#     np.savez_compressed(path, embeddings=embeddings, labels=labels)
#     print(f"📦 Saved embeddings to {path}")
#     return embeddings, labels
