import torch
from torch import optim
from tqdm import tqdm
from data.dataloader import get_dataloader
from pretrain.builder import build_full_model, build_loss
from pretrain.utils import save_checkpoint

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, inputs) if criterion.__code__.co_argcount == 3 else criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, inputs) if criterion.__code__.co_argcount == 3 else criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# def train_model(config, device, experiment_dir, logger, resume=False, best=False):
#     model, encoder = build_full_model(config)
#     model = model.to(device)

#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config['training']['learning_rate'],
#         weight_decay=config['training']['weight_decay']
#     )

#     checkpoint_dir = experiment_dir / "checkpoints"
#     last_ckpt = checkpoint_dir / "last.pt"
#     best_ckpt = checkpoint_dir / "best.pt"

#     best_epoch = 0
#     best_val_loss = float("inf")

#     if best and best_ckpt.exists():
#         print(f"🔍 Loading best model only from: {best_ckpt}")
#         ckpt = torch.load(best_ckpt, map_location=device)
#         model.load_state_dict(ckpt['model_state_dict'])
#         best_epoch = ckpt['epoch']
#         return model, encoder, best_epoch

#     start_epoch = 1
#     if resume and last_ckpt.exists():
#         print(f"🔁 Resuming training from: {last_ckpt}")
#         ckpt = torch.load(last_ckpt, map_location=device)
#         model.load_state_dict(ckpt['model_state_dict'])
#         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         best_val_loss = ckpt['val_loss']
#         start_epoch = ckpt['epoch'] + 1
#         best_epoch = ckpt['epoch']

#     train_loader = get_dataloader(config, split="train")
#     val_loader = get_dataloader(config, split="val")
#     loss_fn = build_loss(config)

#     patience = config['training'].get('patience', 10)
#     patience_counter = 0

#     for epoch in range(start_epoch, config['training']['epochs'] + 1):
#         print(f"Epoch [{epoch}/{config['training']['epochs']}]")
#         train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
#         val_loss = validate_epoch(model, val_loader, loss_fn, device)

#         logger.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
#         print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         is_best = val_loss < best_val_loss
#         if is_best:
#             best_val_loss = val_loss
#             best_epoch = epoch
#             patience_counter = 0
#         else:
#             patience_counter += 1

#         save_checkpoint(model, optimizer, val_loss, epoch, experiment_dir, is_best=is_best)

#         if config['training']['early_stopping'] and patience_counter >= patience:
#             print("🛑 Early stopping triggered.")
#             break

#     return model, encoder, best_epoch


def train_model(config, device, experiment_dir, logger, resume=False, best=False):
    model, encoder = build_full_model(config)
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    checkpoint_dir = experiment_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last.pt"
    best_ckpt = checkpoint_dir / "best.pt"

    best_epoch = 0
    best_val_loss = float("inf")

    if best and best_ckpt.exists():
        print(f"🔍 Loading best model only from: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_epoch = ckpt['epoch']
        return model, encoder, best_epoch, ckpt['val_loss']

    start_epoch = 1
    if resume and last_ckpt.exists():
        print(f"🔁 Resuming training from: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_val_loss = ckpt['val_loss']
        start_epoch = ckpt['epoch'] + 1
        best_epoch = ckpt['epoch']

    train_loader = get_dataloader(config, split="train")
    val_loader = get_dataloader(config, split="val")
    loss_fn = build_loss(config)

    patience = config['training'].get('patience', 10)
    patience_counter = 0

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"Epoch [{epoch}/{config['training']['epochs']}]")
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        logger.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, val_loss, epoch, experiment_dir, is_best=is_best)

        if config['training']['early_stopping'] and patience_counter >= patience:
            print("🛑 Early stopping triggered.")
            break

    return model, encoder, best_epoch, best_val_loss



