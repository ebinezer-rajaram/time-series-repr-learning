from torch.utils.data import DataLoader
from data.datasets import MarketDataset

def get_dataloader(config, split):
    dataset_name = config['data']['dataset']
    kwargs = dict(
        window_size=config['data']['window_size'],
        horizon=config['data']['horizon'],
        split=split,
        transforms=None,
    )
    if dataset_name == 'financial':
        ds = MarketDataset(
            csv_path=config['data']['csv_path'],
            **kwargs
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")

    return DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
