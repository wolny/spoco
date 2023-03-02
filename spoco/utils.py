import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

SUPPORTED_DATASETS = ['cvppp', 'cityscapes', 'h5', 'brightfield']


class GaussianKernel(nn.Module):
    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


def create_optimizer(model, lr, wd=0., betas=(0.9, 0.999)):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    betas = tuple(betas)
    return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)


def create_lr_scheduler(optimizer, patience, mode, factor=0.2):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    return lr_scheduler


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename):
    checkpoint_dir, _ = os.path.split(filename)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.save(state, filename)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(filename, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict',
                    map_location=None):
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def pca_project(embeddings):
    """
    Project embeddings into 3-dim RGB space for visualization purposes

    Args:
        embeddings: ExSpatial embedding tensor

    Returns:
        RGB image
    """
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return img.astype('uint8')


def minmax_norm(img):
    channels = [np.nan_to_num((c - np.min(c)) / np.ptp(c)) for c in img]
    return np.stack(channels, axis=0)
