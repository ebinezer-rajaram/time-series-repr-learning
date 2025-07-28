# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from pathlib import Path
# from tqdm import tqdm
# from datetime import datetime
# import yaml
# import numpy as np

# from utils.config import load_config  # assumed YAML loader
# from data.dataloader import get_dataloader
# from models import build_model
# from models.heads import ForecastingHead, MaskingHead, ContrastiveHead
# from objectives import get_loss
# from utils.metrics.logger import MetricLogger
# from utils.visualisation import PlotLogger
# from utils.metrics import compute_all
# from utils.visualisation import plot_all

# def setup_experiment(config) -> Path:
#     base_dir = Path(config['logging']['log_dir'])
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     experiment_name = f"{config['model']['name']}_{config['objective']['type']}_{timestamp}"
#     experiment_dir = base_dir / experiment_name
#     experiment_dir.mkdir(parents=True, exist_ok=True)

#     with open(experiment_dir / "config.yaml", "w") as f:
#         yaml.safe_dump(config, f)

#     return experiment_dir


# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0

#     for batch in tqdm(dataloader, desc="Training", leave=False):
#         inputs, targets = batch
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)

#         # handle objectives that require extra args (e.g., masking)
#         if criterion.__code__.co_argcount == 3:
#             loss = criterion(outputs, targets, inputs)  # e.g., masking
#         else:
#             loss = criterion(outputs, targets)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)

#     return running_loss / len(dataloader.dataset)


# def validate_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Validation", leave=False):
#             inputs, targets = batch
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)

#             if criterion.__code__.co_argcount == 3:
#                 loss = criterion(outputs, targets, inputs)
#             else:
#                 loss = criterion(outputs, targets)

#             running_loss += loss.item() * inputs.size(0)

#     return running_loss / len(dataloader.dataset)


# def extract_embeddings(model, dataloader, device):
#     model.eval()
#     embeddings = []
#     labels = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             outputs = model(inputs)

#             embeddings.append(outputs.detach().cpu())
#             labels.append(targets)

#     embeddings = torch.cat(embeddings).numpy()  # currently [B, seq_len, dim]
#     labels = torch.cat(labels).numpy()

#     # 🔷 reduce to [B, dim]:
#     embeddings = embeddings.mean(axis=1)       
    
#     if labels.ndim > 1 and labels.shape[1] > 1:
#         labels = labels[:, 0]  # or whatever makes sense semantically
#     labels = labels.squeeze()
 

#     return embeddings, labels



# def main():
#     config = load_config("configs/test.yaml")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     experiment_dir = setup_experiment(config)
    
#     metrics_dir = experiment_dir / "metrics"
#     metrics_dir.mkdir(exist_ok=True)

#     metrics_dir = experiment_dir / "metrics"
#     metrics_dir.mkdir(exist_ok=True)

#     train_logger = MetricLogger(
#         metrics_dir, csv_name="training.csv"
#     )

#     final_logger = MetricLogger(
#         metrics_dir, csv_name="final.csv"
#     )

#     plot_logger = PlotLogger(experiment_dir.parent, experiment_dir.name)

#     train_loader = get_dataloader(config, split="train")
#     val_loader = get_dataloader(config, split="val")

#     encoder = build_model(
#         config['model']['name'],
#         input_dim=config['model']['input_dim'],
#         model_dim=config['model'].get('model_dim', 64),
#         hidden_dim=config['model'].get('hidden_dim', 64),
#         num_layers=config['model'].get('num_layers', 2),
#         num_heads=config['model'].get('num_heads', 4),
#         dim_feedforward=config['model'].get('dim_feedforward', 128),
#         dropout=config['model'].get('dropout', 0.1)
#     ).to(device)

#     obj_type = config['objective']['type'].lower()

#     if obj_type == "forecasting":
#         model = ForecastingHead(
#             encoder,
#             model_dim=config['model']['model_dim'],
#             input_dim=config['model']['input_dim'],
#             pred_len=config['objective']['pred_len']
#         ).to(device)

#     elif obj_type == "masking":
#         model = MaskingHead(
#             encoder,
#             model_dim=config['model']['model_dim'],
#             input_dim=config['model']['input_dim']
#         ).to(device)

#     elif obj_type == "contrastive":
#         model = ContrastiveHead(
#             encoder,
#             model_dim=config['model']['model_dim'],
#             projection_dim=config['objective'].get('projection_dim', 64)
#         ).to(device)

#     else:
#         raise ValueError(f"Unsupported objective: {obj_type}")


#     loss_fn = get_loss(config['objective']['type'], **config['objective'])

#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config['training']['learning_rate'],
#         weight_decay=config['training']['weight_decay']
#     )

#     checkpoint_dir = experiment_dir / "checkpoints"
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)

#     best_val_loss = float("inf")
#     patience_counter = 0

#     for epoch in range(1, config['training']['epochs'] + 1):
#         print(f"Epoch [{epoch}/{config['training']['epochs']}]")

#         train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
#         val_loss = validate_epoch(model, val_loader, loss_fn, device)

#         train_logger.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

#         print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save({'epoch': epoch,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'val_loss': val_loss,
#                     }, checkpoint_dir / "best.pt")
#             print(f"✅ Saved new best checkpoint at epoch {epoch} (val_loss: {val_loss:.4f})")
#         else:
#             patience_counter += 1
#             if config['training']['early_stopping'] and patience_counter >= config['training']['patience']:
#                 print("🛑 Early stopping triggered.")
#                 break


#     embeddings, labels = extract_embeddings(model, val_loader, device)

#     embeddings_path = experiment_dir / "embeddings.npz"
#     np.savez_compressed(
#         embeddings_path,
#         embeddings=embeddings,
#         labels=labels
#     )
#     print(f"📦 Saved embeddings to {embeddings_path}")

#     final_metrics = compute_all(
#         torch.tensor(embeddings),        
#         z_pos=None,                       
#         logits=None,                      
#         labels=torch.tensor(labels)      
#     )
    
#     final_logger.log(final_metrics, step=config['training']['epochs'])

#     # import json
#     # with open(experiment_dir / "final_metrics.json", "w") as f:
#     #     json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)
    
#     plot_all(
#         embeddings=embeddings,
#         labels=labels,
#         metrics=final_metrics,
#         encoder=encoder,
#         plot_logger=plot_logger,
#         experiment_dir=experiment_dir,
#         step=config['training']['epochs'],
#     )

#     train_logger.close()
#     final_logger.close()
#     print("Training complete.")


# if __name__ == "__main__":
#     main()