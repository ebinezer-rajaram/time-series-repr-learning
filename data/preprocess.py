import pandas as pd
import numpy as np
import os
from utils.config import load_config

def preprocess_market(config):
    dataset_name = config['data']['dataset']
    raw_path = config['data'].get('raw_path', f"data/raw/{dataset_name}.csv")
    save_dir = config['data'].get('save_dir', "data/processed/")
    features = config['data'].get('features', ['Open', 'High', 'Low', 'Close', 'Volume'])
    method = config['data'].get('method', 'zscore')

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df = df[features].dropna().copy()

    n = len(df)
    train_split = config['data'].get('train_split', 0.7)
    val_split = config['data'].get('val_split', 0.15)
    test_split = config['data'].get('test_split', 0.15)

    train_end = int(train_split * n)
    val_end = int((train_split + val_split) * n)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    if method == 'zscore':
        mean, std = train.mean(), train.std() + 1e-6
        train_scaled = (train - mean) / std
        val_scaled = (val - mean) / std
        test_scaled = (test - mean) / std
    elif method == 'minmax':
        min_, max_ = train.min(), train.max()
        train_scaled = (train - min_) / (max_ - min_ + 1e-6)
        val_scaled = (val - min_) / (max_ - min_ + 1e-6)
        test_scaled = (test - min_) / (max_ - min_ + 1e-6)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    # restore Date column
    train_scaled = train_scaled.reset_index()
    val_scaled = val_scaled.reset_index()
    test_scaled = test_scaled.reset_index()

    # save
    train_scaled.to_csv(os.path.join(save_dir, f"{dataset_name}_train.csv"), index=False)
    val_scaled.to_csv(os.path.join(save_dir, f"{dataset_name}_val.csv"), index=False)
    test_scaled.to_csv(os.path.join(save_dir, f"{dataset_name}_test.csv"), index=False)

    print(f"✅ Saved processed splits to {save_dir}")
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    preprocess_market(config)
