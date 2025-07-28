from pathlib import Path
from datetime import datetime
import yaml
import csv
import json
import torch

# def setup_experiment(config, experiment_name=None) -> Path:
#     base_dir = Path(config['logging']['log_dir'])

#     if experiment_name is None:
#         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         timestamp = "test"
#         experiment_name = f"{config['model']['name']}_{config['objective']['type']}_{timestamp}"

#     experiment_dir = base_dir / experiment_name
#     experiment_dir.mkdir(parents=True, exist_ok=True)

#     with open(experiment_dir / "config.yaml", "w") as f:
#         yaml.safe_dump(config, f)

#     return experiment_dir


def setup_experiment(config, experiment_name=None) -> Path:
    model = config['model']['name']
    objective = config['objective']['type']
    base_dir = Path(config['logging']['log_dir'])
    group_dir = base_dir / f"{model}_{objective}"
    group_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([p for p in group_dir.glob("exp_*") if p.is_dir()])
    run_id = len(existing) + 1
    run_name = f"exp_{run_id:03}"

    experiment_dir = group_dir / run_name
    experiment_dir.mkdir(parents=True, exist_ok=False) 

    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    index_path = group_dir / "index.csv"
    index_exists = index_path.exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not index_exists:
            writer.writerow(["model", "objective", "run_id", "path", "val_loss"])
        writer.writerow([model, objective, run_name, str(experiment_dir), ""])


    return experiment_dir



def update_best_run_marker(experiment_dir: Path, val_loss: float):
    group_dir = experiment_dir.parent
    best_file = group_dir / "best.txt"
    index_path = group_dir / "index.csv"

    # 1. Update index.csv val_loss
    run_id = experiment_dir.name
    model, objective = group_dir.name.split("_", 1)

    updated_rows = []
    with open(index_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == model and row["objective"] == objective and row["run_id"] == run_id:
                row["val_loss"] = f"{val_loss:.6f}"
            updated_rows.append(row)

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "objective", "run_id", "path", "val_loss"])
        writer.writeheader()
        writer.writerows(updated_rows)

    # 2. Write best.txt
    with open(best_file, "w") as f:
        json.dump({"run_id": run_id, "val_loss": val_loss}, f, indent=2)

    print(f"🏆 Updated best.txt → {run_id} (val_loss={val_loss:.4f})")




def save_checkpoint(model, optimizer, val_loss, epoch, experiment_dir, is_best=False):
    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    torch.save(state, ckpt_dir / "last.pt")  # always save latest
    if is_best:
        torch.save(state, ckpt_dir / "best.pt")
        print(f"✅ Saved new best checkpoint (val_loss: {val_loss:.4f})")

