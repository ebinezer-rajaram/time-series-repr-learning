# time-series-repr-learning
Self-supervised Transformers for Multivariate Time Series Representation Learning


## 🧪 Pretraining (Self-Supervised)

This module pretrains a transformer (or other encoder) on multivariate time series using a self-supervised objective like masked reconstruction or contrastive learning.

---

### 🧠 Running Pretraining

Use the following command:

```bash
python -m pretrain.run --config configs/<your_config>.yaml
```

---

### 🔧 CLI Options

| Flag | Description |
|------|-------------|
| `--config` | Path to config file (required) |
| `--resume` | Resume training from the last checkpoint of a specific run |
| `--best` | Skip training and evaluate the best checkpoint only |
| `--experiment-name` | Manually specify experiment path (e.g. `transformer_forecasting/exp_003`) |
| `--use-best` | Automatically resolve to best checkpoint across all runs for the model/objective group |
| `--latest` | Automatically resolve to the most recent `exp_XXX` run in the group |

---

### ✅ Example Workflows

#### 🚀 Train from scratch

```bash
python -m pretrain.run --config configs/ts_transformer.yaml
```

This creates a new directory like:

```
runs/transformer_forecasting/exp_001/
```

---

#### 🔁 Resume a previous run

```bash
python -m pretrain.run --config configs/ts_transformer.yaml \
  --resume --experiment-name transformer_forecasting/exp_002
```

Loads from `checkpoints/last.pt` and continues training.

---

#### 🧪 Evaluate a specific experiment’s best checkpoint

```bash
python -m pretrain.run --config configs/ts_transformer.yaml \
  --best --experiment-name transformer_forecasting/exp_003
```

Skips training. Loads `checkpoints/best.pt` and runs embedding extraction, metric logging, and plot generation.

---

#### 🏆 Evaluate the best overall run (across all experiments)

```bash
python -m pretrain.run --config configs/ts_transformer.yaml \
  --best --use-best
```

Follows the `best/` symlink inside the model-objective group directory.

---

#### 🔂 Resume the most recent run

```bash
python -m pretrain.run --config configs/ts_transformer.yaml \
  --resume --latest
```

Resumes training from the most recently created `exp_XXX/` under the correct group.

---

### 📁 Output Directory Structure

All outputs go to:

```
runs/{model}_{objective}/exp_XXX/
```

Each run contains:

- `checkpoints/`
  - `last.pt` — last saved checkpoint
  - `best.pt` — best validation checkpoint
- `metrics/`
  - `training.csv`, `final.csv`
- `plots/`
- `embeddings.pt`
- `config.yaml`

Additionally:

- The **best run** is symlinked at: `runs/{model}_{objective}/best/`
- A CSV index of all runs is logged to: `runs/index.csv`

---

### 📌 Notes

- If `--resume` or `--best` is passed, you **must** also pass `--experiment-name` or use `--use-best` / `--latest`.
- The config file controls model architecture, loss type, dataset, and training hyperparameters.

---
