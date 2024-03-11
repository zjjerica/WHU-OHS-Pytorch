# Dataset construction based on public hyperspectral dataset
import os
import torch
from torch.utils import data
import numpy as np

# Patch-based dataset, i.e., overlapping patches are generated using a window of small size as network input.
class Hyper_Dataset_patch(data.Dataset):
    def __init__(self, image, gt, windowsize=15, use_3D_input=False, channel_last=False):
        self.image = image.astype(np.float)
        self.gt = gt
        self.windowsize = windowsize
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last

        bands = image.shape[2]
        for i in range(bands):
            temp = self.image[:, :, i]
            mean = np.mean(temp)
            std = np.std(temp)
            if (std == 0):
                self.image[:, :, i] = 0
            else:
                self.image[:, :, i] = (temp - mean) / std

        x_pos, y_pos = np.nonzero(self.gt)
        p = self.windowsize // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < self.image.shape[0] - p and y > p and y < self.image.shape[1] - p
            ]
        )
        np.random.shuffle(self.indices)
        self.labels = [self.gt[x, y] for x, y in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x_center, y_center = self.indices[index]
        label = self.labels[index]

        shift = int((self.windowsize - 1) / 2)
        data = self.image[x_center - shift:x_center + shift + 1, y_center - shift:y_center + shift + 1, :]
        data = torch.tensor(data, dtype=torch.float)

        if not self.channel_last:
            data = data.permute(2, 0, 1)

        if self.use_3D_input:
            data = data.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long) - 1

        return data, label