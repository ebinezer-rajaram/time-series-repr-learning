from torch.utils.data import DataLoader
from data.datasets import DATASETS


def get_dataloader(config, split, transforms=None):
    dataset_name = config['data']['dataset']
    dataset_cls = DATASETS.get(dataset_name)

    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    kwargs = dict(
        window_size=config['data']['seq_len'],
        horizon=config['data']['pred_len'],
        split=split,
        transforms=transforms,
    )

    if dataset_name == 'synthetic':
        kwargs['params'] = config['data'].get('params', {})
    else:
        suffix = f"{split}.csv"
        kwargs['csv_path'] = f"data/processed/{dataset_name}_{suffix}"

    dataset = dataset_cls(**kwargs)

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],  # prefer batch_size from 'data' section
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )

    return loader
