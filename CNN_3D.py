# 3D-CNN network
# Y. Chen, H. Jiang, C. Li, X. Jia, and P. Ghamisi, “Deep Feature Extraction and Classification of Hyperspectral Images
# Based on Convolutional Neural Networks,” IEEE Trans. Geosci. Remote Sens., vol. 54, pp. 6232–6251, 2016.

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_3D(nn.Module):
    def __init__(self, input_features, n_classes):
        super(CNN_3D, self).__init__()
        self.input_features = input_features
        self.n_classes = n_classes

        self.conv3d_1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(6, 5, 5), padding=(0, 2, 2))
        self.conv3d_2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(6, 5, 5), padding=(0, 2, 2))

        self.final_size = self.get_final_size()
        self.final_conv = nn.Conv2d(in_channels=self.final_size, out_channels=self.n_classes, kernel_size=1)

    def get_final_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_features, 512, 512)
            x = self.conv3d_1(x)
            x = self.conv3d_2(x)
            final_size = x.size(1) * x.size(2)
        return final_size

    def forward(self, x):
        x = self.conv3d_1(x)
        x = F.relu(x)
        x = self.conv3d_2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.final_conv(x)

        return x