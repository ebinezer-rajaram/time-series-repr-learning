from torch.utils.data import DataLoader
from data import DATASETS
import os

def get_dataloader(config, split, transforms=None):
    dataset_name = config['data']['dataset']
    dataset_cls = DATASETS.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    split_ratios = (
        config['data'].get('train_split', 0.7),
        config['data'].get('val_split', 0.15),
        config['data'].get('test_split', 0.15)
    )

    kwargs = dict(
        window_size=config['data']['seq_len'],
        horizon=config['data']['pred_len'],
        split=split,
        transforms=transforms,
        split_ratios=split_ratios,
        csv_path=f"data/processed/{dataset_name}_full.csv"
    )

    dataset = dataset_cls(**kwargs)

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(split == 'train'),
        drop_last=False,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader
