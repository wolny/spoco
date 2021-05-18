import collections

import torch
from torch.utils.data import DataLoader

from spoco.datasets.cvppp import CVPPP2017Dataset


def create_train_val_loaders(ds_name, ds_path, batch_size, num_workers, instance_ratio, random_seed):
    if ds_name == 'cvppp':
        train_datasets = CVPPP2017Dataset(ds_path, 'train', instance_ratio=instance_ratio, random_seed=random_seed)
        val_datasets = CVPPP2017Dataset(ds_path, 'val')

    # TODO: add remaining dataset

    print(f'Number of workers for train/val dataloader: {num_workers}')
    print(f'Batch size for train/val loader: {batch_size}')
    if torch.cuda.device_count() > 1:
        print(
            f'{torch.cuda.device_count()} GPUs available. '
            f'Increasing batch_size: {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return (
        DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        # don't shuffle during validation
        DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )


def dsb_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
