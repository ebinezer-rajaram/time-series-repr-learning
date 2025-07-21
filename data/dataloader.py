from torch.utils.data import DataLoader
from data.datasets import DATASETS


def get_dataloader(config, split, transforms=None):
    dataset_name = config['data']['dataset']
    dataset_cls = DATASETS.get(dataset_name)

    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_name} not implemented.")

    kwargs = dict(
        window_size=config['data']['window_size'],
        horizon=config['data']['horizon'],
        split=split,
        transforms=transforms,
    )

    if dataset_name == 'synthetic':
        kwargs['params'] = config['data'].get('params', {})
    else:
        kwargs['csv_path'] = config['data']['csv_path']

    dataset = dataset_cls(**kwargs)

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )

    return loader
