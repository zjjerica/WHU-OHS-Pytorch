# Dataset construction based on WHU-OHS
import os
import torch
from torch.utils import data
import numpy as np
from osgeo import gdal

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class WHU_OHS_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list, use_3D_input=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.use_3D_input = use_3D_input

    # Statistics of samples of each class in the dataset
    def sample_stat(self):
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = gdal.Open(label_file, gdal.GA_ReadOnly)
            label = label.ReadAsArray()
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class = sample_per_class + count

        return sample_per_class

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()

        # The image patches were normalized and scaled by 10000 to reduce storage cost
        image = torch.tensor(image, dtype=torch.float) / 10000.0

        if(self.use_3D_input):
            image = image.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long) - 1

        return image, label, name











