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
            x = data[i : i + self.window_size]
            y = data[i + self.window_size : i + self.window_size + self.horizon]
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    def train_val_test_split(self, data):
        n = len(data)
        if self.split == 'train':
            return data[:int(0.7*n)]
        elif self.split == 'val':
            return data[int(0.7*n):int(0.85*n)]
        else:
            return data[int(0.85*n):]


class MarketDataset(BaseTimeSeriesDataset):
    def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
        super().__init__(window_size, horizon, split, transforms)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

        data = (data - data.mean(0)) / (data.std(0) + 1e-6)

        data = self.train_val_test_split(data)
        self.X, self.Y = self.create_windows(data)


# class ETTH1Dataset(BaseTimeSeriesDataset):
#     def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
#         super().__init__(window_size, horizon, split, transforms)
#         df = pd.read_csv(csv_path)
#         data = df.iloc[:, 1:].values

#         data = (data - data.mean(0)) / (data.std(0) + 1e-6)

#         data = self.train_val_test_split(data)
#         self.X, self.Y = self.create_windows(data)


# class ECGDataset(BaseTimeSeriesDataset):
#     def __init__(self, csv_path, window_size, horizon, split='train', transforms=None):
#         super().__init__(window_size, horizon, split, transforms)
#         df = pd.read_csv(csv_path)
#         data = df.values

#         data = (data - data.mean()) / (data.std() + 1e-6)

#         data = self.train_val_test_split(data)
#         self.X, self.Y = self.create_windows(data)


# class SyntheticDataset(BaseTimeSeriesDataset):
#     def __init__(self, params, window_size, horizon, split='train', transforms=None):
#         super().__init__(window_size, horizon, split, transforms)
#         np.random.seed(123) # for reproducibility
#         T = params.get("length", 10000)
#         freq = params.get("freq", 0.01)
#         n_features = params.get("n_features", 1)

#         x = np.arange(T)
#         data = np.zeros((T, n_features))
#         for i in range(n_features):
#             data[:, i] = np.sin(2 * np.pi * freq * x + i) + 0.1 * np.random.randn(T)

#         data = self.train_val_test_split(data)
#         self.X, self.Y = self.create_windows(data)
