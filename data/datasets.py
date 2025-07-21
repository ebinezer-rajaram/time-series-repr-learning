import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BaseTimeSeriesDataset(Dataset):
    def __init__(self, window_size, horizon, split='train', transforms=None):
        self.window_size = window_size
        self.horizon = horizon
        self.split = split
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def create_windows(self, data):
        X, Y = [], []
        for i in range(len(data) - self.window_size - self.horizon):
            x = data[i: i + self.window_size]
            y = data[i + self.window_size: i + self.window_size + self.horizon]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def train_val_test_split(self, data):
        n = len(data)
        if self.split == 'train':
            return data[:int(0.7 * n)]
        elif self.split == 'val':
            return data[int(0.7 * n): int(0.85 * n)]
        else:
            return data[int(0.85 * n):]

    def __repr__(self):
        return f"{self.__class__.__name__}(split={self.split}, N={len(self)})"


class MarketDataset(BaseTimeSeriesDataset):
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
        super().__init__(window_size, horizon, split, transforms)

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        data = df.values

        data = self.train_val_test_split(data)
        self.X, self.Y = self.create_windows(data)


class ETTH1Dataset(BaseTimeSeriesDataset):
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
        super().__init__(window_size, horizon, split, transforms)

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data = df.values

        data = self.train_val_test_split(data)
        self.X, self.Y = self.create_windows(data)


class ECGDataset(BaseTimeSeriesDataset):
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
        super().__init__(window_size, horizon, split, transforms)

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data = df.values

        data = self.train_val_test_split(data)
        self.X, self.Y = self.create_windows(data)


class SyntheticDataset(BaseTimeSeriesDataset):
    def __init__(self, csv_path=None, window_size=50, horizon=10, split='train', transforms=None):
        super().__init__(window_size, horizon, split, transforms)

        if csv_path:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            full_data = df.values
        else:
            # fallback: generate synthetic if no file
            T, freq, n_features = 10000, 0.01, 1
            x = np.arange(T)
            full_data = np.zeros((T, n_features))
            for i in range(n_features):
                full_data[:, i] = np.sin(2 * np.pi * freq * x + i) + 0.1 * np.random.randn(T)

        data = self.train_val_test_split(full_data)
        self.X, self.Y = self.create_windows(data)


DATASETS = {
    'financial': MarketDataset,
    'etth1': ETTH1Dataset,
    'ecg': ECGDataset,
    'synthetic': SyntheticDataset,
}
