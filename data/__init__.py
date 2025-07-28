from .datasets import MarketDataset, ETTH1Dataset, ECGDataset, SyntheticDataset

DATASETS = {
    'spy': MarketDataset,
    'etth1': ETTH1Dataset,
    'ecg': ECGDataset,
    'synthetic': SyntheticDataset,
}
