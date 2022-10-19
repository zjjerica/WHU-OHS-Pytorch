# 1D-CNN network
# W. Hu, Y. Huang, L. Wei, F. Zhang, and H. Li, Deep Convolutional Neural Networks for Hyperspectral Image
# Classification, J. Sensors, vol. 2015, p. 12, 2015.

import math
import torch
import torch.nn as nn

class CNN_1D(nn.Module):
    def get_final_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_features, 512, 512)
            x = self.conv1d(x)
            x = self.pool1d(x)
            final_size = x.size(1) * x.size(2)
        return final_size

    def __init__(self, input_features, n_classes):
        super(CNN_1D, self).__init__()
        self.input_features = input_features
        self.n_classes = n_classes

        self.kernel_size = math.floor(self.input_features / 9)
        self.pool_size = math.ceil((self.input_features - self.kernel_size + 1) / 30)
        if(self.pool_size == 0):
            self.pool_size = self.pool_size + 1

        self.conv1d = nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(self.kernel_size, 1, 1))
        self.pool1d = nn.MaxPool3d(kernel_size=(self.pool_size, 1, 1))
        self.final_size = self.get_final_size()
        self.final_conv_1 = nn.Conv2d(in_channels=self.final_size, out_channels=100, kernel_size=1)
        self.final_conv_2 = nn.Conv2d(in_channels=100, out_channels=self.n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.tanh(x)
        x = self.pool1d(x)
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        x = self.final_conv_1(x)
        x = torch.tanh(x)
        x = self.final_conv_2(x)

        return x
