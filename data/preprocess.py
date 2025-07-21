import pandas as pd
import numpy as np
import os

def preprocess_market(
    raw_path="data/raw/spy.csv",
    save_dir="data/processed/",
    features=None,
    method='zscore'
):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    if features is None:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']

    df = df[features].dropna().copy()

    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

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

    # restore Date column for clarity
    train_scaled = train_scaled.reset_index()
    val_scaled = val_scaled.reset_index()
    test_scaled = test_scaled.reset_index()

    train_scaled.to_csv(os.path.join(save_dir, "spy_train.csv"), index=False)
    val_scaled.to_csv(os.path.join(save_dir, "spy_val.csv"), index=False)
    test_scaled.to_csv(os.path.join(save_dir, "spy_test.csv"), index=False)

    print(f"✅ Saved processed splits to {save_dir}")
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    preprocess_market()
