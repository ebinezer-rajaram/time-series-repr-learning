import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def make_split_dataset(df, split, window_size, horizon, split_ratios=(0.7, 0.15, 0.15), transforms=None):
    data = df.values
    X, Y = BaseTimeSeriesDataset.create_windows(data, window_size, horizon)
    n = len(X)

    train_end = int(split_ratios[0] * n)
    val_end = train_end + int(split_ratios[1] * n)

    if split == 'train':
        X_split, Y_split = X[:train_end], Y[:train_end]
    elif split == 'val':
        X_split, Y_split = X[train_end:val_end], Y[train_end:val_end]
    else:
        X_split, Y_split = X[val_end:], Y[val_end:]

    return BaseTimeSeriesDataset(X_split, Y_split, transforms=transforms)

class BaseTimeSeriesDataset(Dataset):
    def __init__(self, X, Y, transforms=None):
        self.X = X
        self.Y = Y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def create_windows(data, window_size, horizon):
        X, Y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            x = data[i:i + window_size]
            y = data[i + window_size:i + window_size + horizon]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)


class MarketDataset:
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15)):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.ds = make_split_dataset(df, split, window_size, horizon, split_ratios, transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class ETTH1Dataset:
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15)):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.ds = make_split_dataset(df, split, window_size, horizon, split_ratios, transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class ECGDataset:
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15)):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        self.ds = make_split_dataset(df, split, window_size, horizon, split_ratios, transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class SyntheticDataset:
    def __init__(self, csv_path=None, window_size=50, horizon=10, split='train', transforms=None, split_ratios=(0.7, 0.15, 0.15)):
        if csv_path:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            # fallback: generate synthetic signal
            T, freq, n_features = 10000, 0.01, 1
            x = np.arange(T)
            arr = np.zeros((T, n_features))
            for i in range(n_features):
                arr[:, i] = np.sin(2 * np.pi * freq * x + i) + 0.1 * np.random.randn(T)
            df = pd.DataFrame(arr)

        self.ds = make_split_dataset(df, split, window_size, horizon, split_ratios, transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

