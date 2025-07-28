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
    train_end = int(train_split * n)
    train = df.iloc[:train_end]

    if method == 'zscore':
        mean, std = train.mean(), train.std() + 1e-6
        df_scaled = (df - mean) / std
    elif method == 'minmax':
        min_, max_ = train.min(), train.max()
        df_scaled = (df - min_) / (max_ - min_ + 1e-6)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    save_path = os.path.join(save_dir, f"{dataset_name}_full.csv")
    df_scaled.to_csv(save_path)
    print(f"\u2705 Saved normalized full dataset to {save_path}")
    return df_scaled

if __name__ == "__main__":
    config = load_config("configs/default.yaml")
    preprocess_market(config)
