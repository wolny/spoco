import glob
import os
import random

import numpy as np
import imageio
import torch
from PIL import Image
from skimage.measure import label
from torchvision.transforms import transforms

from spoco.transforms import GaussianBlur, ImgNormalize, LabelToTensor


EXTENDED_TRANSFORM = transforms.Compose(
    [
        transforms.RandomApply([
            GaussianBlur([.1, 2.])
        ], p=0.5),
        transforms.ToTensor(),
        ImgNormalize()
    ]
)

DOWNSIZE = size = (512, 672)

SPOCO_TEST = transforms.Compose(
    [
        transforms.Resize(DOWNSIZE),
        GaussianBlur([.1, 2.]),
        transforms.ToTensor(),
        ImgNormalize()
    ]
)


class BrightfieldDataset:
    def __init__(self, root_dir, phase):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        self.raw_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(DOWNSIZE, scale=(0.7, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]
        )

        self.phase = phase
        self.images, self.paths = self._load_files(os.path.join(root_dir, phase))
        assert len(self.images) > 0

        self.train_label_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(DOWNSIZE, scale=(0.7, 1.), interpolation=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                LabelToTensor(False),
            ]
        )

        self.val_label_transform = transforms.Compose(
            [
                transforms.Resize(DOWNSIZE, interpolation=Image.NEAREST),
                LabelToTensor(False),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(DOWNSIZE),
                transforms.ToTensor(),
                ImgNormalize()
            ]
        )

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase == 'train':
            lbl = self._load_lbl(idx)
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            # make it work with torch vision v0.8.0+
            torch.manual_seed(seed)
            img = self.raw_transform(img)
            random.seed(seed)
            # make it work with torch vision v0.8.0+
            torch.manual_seed(seed)
            lbl = self.train_label_transform(lbl)

            img1 = EXTENDED_TRANSFORM(img)
            img2 = EXTENDED_TRANSFORM(img)
            return img1, img2, lbl
        elif self.phase == 'val':
            lbl = self._load_lbl(idx)
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            img = self.test_transform(img)
            random.seed(seed)
            mask = self.val_label_transform(lbl)
            return img, img, mask

        else:
            img1 = self.test_transform(img)
            img2 = SPOCO_TEST(img)
            return img1, img2, self.paths[idx]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_files(root_dir):
        # we only load raw or label images
        files_data = []
        paths = list(glob.glob(os.path.join(root_dir, '*.tif')))
        for file in paths:
            img = imageio.imread(file)
            img = np.array(img)
            img = Image.fromarray(img, mode='L')
            files_data.append(img)

        return files_data, paths

    def _load_lbl(self, idx):
        path = self.paths[idx]
        path, _ = os.path.splitext(path)
        path = path + '_label.png'
        img = imageio.imread(path)
        img = img[..., 0]

        mask = img == 0
        lbl = label(mask).astype('uint8')
        return Image.fromarray(lbl, mode='L')
