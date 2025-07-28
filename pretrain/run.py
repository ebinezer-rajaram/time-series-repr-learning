import torch
import argparse
import os
import json
from pathlib import Path
from utils.config import load_config
from pretrain.utils import setup_experiment, update_best_run_marker
from pretrain.train_loop import train_model
from pretrain.embeddings import extract_and_save_embeddings
from utils.metrics.logger import MetricLogger
from utils.visualisation import PlotLogger
from utils.metrics import compute_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--best", action="store_true", help="Skip training and eval best checkpoint only")
    parser.add_argument("--experiment-name", type=str, default=None, help="Manually specify experiment path")
    parser.add_argument("--use-best", action="store_true", help="Use best checkpoint from model/objective group")
    parser.add_argument("--latest", action="store_true", help="Use latest exp_XXX from model/objective group")
    args = parser.parse_args()

    config = load_config(args.config)
    model = config['model']['name']
    objective = config['objective']['type']
    group_dir = Path(config['logging']['log_dir']) / f"{model}_{objective}"

    if args.use_best:
        best_path = group_dir / "best.txt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best.txt found in {group_dir}")
        with open(best_path) as f:
            best = json.load(f)
        args.experiment_name = f"{model}_{objective}/{best['run_id']}"

    elif args.latest:
        exps = sorted([p.name for p in group_dir.glob("exp_*") if p.is_dir()])
        if not exps:
            raise ValueError(f"No experiments found under {group_dir}")
        args.experiment_name = f"{model}_{objective}/{exps[-1]}"

    experiment_dir = setup_experiment(config, experiment_name=args.experiment_name)
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    train_logger = MetricLogger(metrics_dir, csv_name="training.csv")
    final_logger = MetricLogger(metrics_dir, csv_name="final.csv")
    plot_logger = PlotLogger(experiment_dir.parent, experiment_dir.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoder, best_epoch, best_val_loss = train_model(
        config, device, experiment_dir, train_logger, resume=args.resume, best=args.best
    )

    if not args.best:
        update_best_run_marker(experiment_dir, best_val_loss)

    embeddings, labels = extract_and_save_embeddings(model, config, device, experiment_dir)

    final_metrics = compute_all(torch.tensor(embeddings), labels=torch.tensor(labels), z_pos=None, logits=None)
    final_logger.log(final_metrics, step=best_epoch)

    plot_logger.all(
        embeddings=embeddings,
        labels=labels,
        metrics=final_metrics,
        encoder=encoder,
        step=best_epoch,
        logger=final_logger
    )

    train_logger.close()
    final_logger.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
