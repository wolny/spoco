import glob
import os
import random
import time
from itertools import chain

import h5py
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose
from tqdm import tqdm

from spoco.datasets.transforms import get_augmentation, AdditiveGaussianNoise, AdditivePoissonNoise, GaussianBlur3D
from spoco.transforms import ToTensor

EXTENDED_TRANSFORM = Compose([
    GaussianBlur3D(execution_probability=0.25),
    AdditiveGaussianNoise(np.random.RandomState(), scale=(0.0, 0.5), execution_probability=0.5),
    AdditivePoissonNoise(np.random.RandomState(), execution_probability=0.25),
    ToTensor(expand_dims=True)
])

class PatchGenerator:
    def __init__(self, raw_shape, label_shape, patch_shape, stride_shape, volume_multiplier=4):
        self.patch_shape = tuple(patch_shape)
        # if stride_shape is None, take random patches
        if stride_shape is not None:
            stride_shape = tuple(stride_shape)

        self.raw = self._build_slices(raw_shape, patch_shape, stride_shape)

        self.label = None
        if label_shape is not None:
            self.label = self._build_slices(label_shape, patch_shape, stride_shape)
            assert len(self.raw) == len(self.label)

        self.volume_multiplier = volume_multiplier

    def _build_slices(self, ds_shape, patch_shape, stride_shape):
        if stride_shape is None:
            # always use the same RNG to avoid surprises
            random_state = np.random.RandomState(47)
            return self._build_random_slices(ds_shape, patch_shape, random_state)
        else:
            return self._build_strided_slices(ds_shape, patch_shape, stride_shape)

    def _build_strided_slices(self, ds_shape, patch_shape, stride_shape):
        slices = []
        if len(ds_shape) == 4:
            in_channels, shape_z, shape_y, shape_x = ds_shape
        else:
            shape_z, shape_y, shape_x = ds_shape

        patch_z, patch_y, patch_x = patch_shape
        stride_z, stride_y, stride_x = stride_shape
        z_steps = PatchGenerator._gen_indices(shape_z, patch_z, stride_z)
        for z in z_steps:
            y_steps = PatchGenerator._gen_indices(shape_y, patch_y, stride_y)
            for y in y_steps:
                x_steps = PatchGenerator._gen_indices(shape_x, patch_x, stride_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + patch_z),
                        slice(y, y + patch_y),
                        slice(x, x + patch_x)
                    )
                    if len(ds_shape) == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(max_shape, patch, stride):
        assert max_shape >= patch, 'Sample size has to be bigger than the patch size'
        for j in range(0, max_shape - patch + 1, stride):
            yield j
        if j + patch < max_shape:
            yield max_shape - patch

    def _build_random_slices(self, ds_shape, patch_shape, random_state):
        slices = []
        patch_z, patch_y, patch_x = patch_shape
        if len(ds_shape) == 4:
            in_channels, shape_z, shape_y, shape_x = ds_shape
        else:
            shape_z, shape_y, shape_x = ds_shape

        # cover the volume `volume_multiplier` times with random patches
        volume_size = self.volume_multiplier * shape_z * shape_y * shape_x
        patch_size = patch_z * patch_y * patch_x

        while volume_size > 0:
            volume_size -= patch_size
            z = random_state.randint(0, shape_z - patch_z)
            y = random_state.randint(0, shape_y - patch_y)
            x = random_state.randint(0, shape_x - patch_x)

            slice_idx = (
                slice(z, z + patch_z),
                slice(y, y + patch_y),
                slice(x, x + patch_x)
            )
            if len(ds_shape) == 4:
                slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)

        return slices

    @property
    def patch_count(self):
        return len(self.raw)


def create_h5_dataset(args, phase, seed=None):
    ds_path = os.path.join(args.ds_path, phase)
    assert os.path.isdir(ds_path)

    # load files
    file_paths = []
    iters = [glob.glob(os.path.join(ds_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
    for fp in chain(*iters):
        file_paths.append(fp)

    if seed is None:
        seed = int(time.time())

    # create datasets
    datasets = []
    for file_path in file_paths:
        print(f'Loading a {phase} set from: {file_path}')

        with h5py.File(file_path, 'r') as f:
            if args.global_norm:
                # compute mean/std globally
                mean = np.mean(f['raw'])
                std = np.std(f['raw'])
            else:
                mean = None
                std = None

            raw_augmentation = get_augmentation('spoco', 'raw', seed, phase, mean=mean, std=std)
            label_augmentation = None
            if phase in ['train', 'val']:
                label_augmentation = get_augmentation('spoco', 'label', seed, phase)

            raw_shape = f['raw'].shape
            if phase in ['train', 'val']:
                label_shape = f['label'].shape
            else:
                label_shape = None

        patch_shape = args.patch_size
        stride_shape = args.stride_size

        patch_generator = PatchGenerator(raw_shape, label_shape, patch_shape, stride_shape)

        datasets.append(
            HDF5Dataset(file_path, phase, patch_generator, raw_augmentation, label_augmentation, spoco=args.spoco)
        )

    if len(datasets) == 1:
        return datasets[0]

    return ConcatDataset(datasets)


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path, phase, patch_generator, raw_transformer, label_transformer=None, spoco=False,
                 patch_halo=None, raw_internal_path='raw', label_internal_path='label', min_label_ratio=None):
        assert phase in ['train', 'val', 'test']
        self.file_path = file_path
        self.phase = phase
        self.patch_generator = patch_generator
        self.raw_transformer = raw_transformer
        self.label_transformer = label_transformer
        self.spoco = spoco
        self.patch_halo = patch_halo
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path

        print(f'Number of patches:', patch_generator.patch_count)
        if min_label_ratio is not None:
            assert 0 < min_label_ratio <= 1
            print('Filtering patches with little signal...')
            label, raw = self._filter_patches(min_label_ratio, patch_generator)
            # save filtered patches
            patch_generator.raw = raw
            patch_generator.label = label
            print(f'Number of patches after filtering:', patch_generator.patch_count)


        with h5py.File(self.file_path, 'r') as f:
            self.raw = f[self.raw_internal_path][:]
            self.label = f[self.label_internal_path][:]

    def _filter_patches(self, min_label_ratio, patch_generator):
        raw = []
        label = []
        with h5py.File(self.file_path, 'r') as f:
            for rp, lp in tqdm(zip(patch_generator.raw, patch_generator.label), total=len(patch_generator.raw)):
                if rp is None:
                    # include empty patches
                    raw.append(rp)
                    label.append(lp)
                    continue

                lbl = f[self.label_internal_path][lp]
                non_zero_count = (lbl > 0).sum()
                if non_zero_count >= min_label_ratio * lbl.size:
                    # include patches with less than min_label_ratio of background
                    raw.append(rp)
                    label.append(lp)
                else:
                    # include empty patches sometimes
                    if non_zero_count == 0 and random.random() < 0.01:
                        raw.append(rp)
                        label.append(lp)
        return label, raw

    def _raw_label_patch(self, idx):
        raw_idx = self.patch_generator.raw[idx]
        if self.phase == 'test':
            label_idx = None
        else:
            label_idx = self.patch_generator.label[idx]

        raw_patch = self.raw[raw_idx]
        if label_idx is None:
            lbl_patch = None
        else:
            lbl_patch = self.label[label_idx]

        return raw_patch, lbl_patch

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        if self.phase == 'test':
            # get the raw data patch for a given slice
            raw, _ = self._raw_label_patch(idx)
            raw_idx = self.patch_generator.raw[idx]
            # squeeze any singleton dimensions
            raw = np.squeeze(raw)
            # pad if necessary
            raw = self.pad(raw)
            # augment
            raw_patch = self.raw_transformer(raw)
            return raw_patch, raw_idx
        else:
            # get the raw and label patches for a given slice
            raw, label = self._raw_label_patch(idx)
            # squeeze any singleton dimensions
            raw = np.squeeze(raw)
            label = np.squeeze(label)
            # augment
            raw_patch = self.raw_transformer(raw)
            label_patch = self.label_transformer(label)

            if self.spoco:
                raw1 = raw2 = raw_patch
                if self.phase == 'train':
                    raw1 = EXTENDED_TRANSFORM(raw_patch)
                    raw2 = EXTENDED_TRANSFORM(raw_patch)

                return raw1, raw2, label_patch

            # return the transformed raw and label patches
            return raw_patch, label_patch

    def __len__(self):
        return self.patch_generator.patch_count

    def pad(self, patch):
        if self.patch_halo is None:
            return patch

        if len(self.patch_halo) == 2:
            y, x = self.patch_halo
            pad_width = ((y, y), (x, x))
            if patch.ndim == 3:
                # pad each channel
                channels = [np.pad(c, pad_width=pad_width, mode='reflect') for c in patch]
                volume = np.stack(channels)
            else:
                volume = np.pad(patch, pad_width=pad_width, mode='reflect')
        else:
            z, y, x = self.patch_halo
            pad_width = ((z, z), (y, y), (x, x))
            if patch.ndim == 4:
                # pad each channel
                channels = [np.pad(c, pad_width=pad_width, mode='reflect') for c in patch]
                volume = np.stack(channels)
            else:
                volume = np.pad(patch, pad_width=pad_width, mode='reflect')

        return volume
