# WHU-OHS-Pytorch

**Paper:** 
J. Li, X. Huang, and L. Tu, “WHU-OHS : A benchmark dataset for large-scale Hersepctral Image classification,” Int. J. Appl. Earth Obs. Geoinf., vol. 113, no. September, p. 103022, 2022, doi: 10.1016/j.jag.2022.103022. [[Link]](https://www.sciencedirect.com/science/article/pii/S1569843222002102)

**Dataset download:** http://irsip.whu.edu.cn/resources/WHU_OHS_show.php

## Dateset Introduction
The WHU-OHS dataset is made up of 42 OHS satellite images acquired from more than 40 different locations in China (Fig. 1). The imagery has a spatial resolution of 10 m (nadir) and a swath width of 60 km (nadir). There are 32 spectral channels ranging from the visible to near-infrared range, with an average spectral resolution of 15 nm. We cropped each image into 512 × 512 pixels with a stride of 32. There are 4822, 513, and 2460 sub-images in the training, validation, and test sets, respectively.

![](Dataset_introduction.png)

<p align='center'>Fig. 1. Left: The geographical locations of the 42 images in the WHU-OHS dataset. Right: Examples of local OHS parcels (true-color compositions with R: 670 nm; G: 566 nm; B: 480 nm) and their corresponding reference labels.

## Dataset Format
The dataset was organized in the format shown in Fig. 2.

![](Dataset_format.png)

<p align='center'>Fig. 2. Data organization of the WHU-OHS dataset.

## Code
Pytorch toolbox for large-scale hyperspectral image classification using WHU-OHS dataset. The deep network models will be updated continuously.

Deep network models:

**1D-CNN**: W. Hu, Y. Huang, L. Wei, F. Zhang, and H. Li, “Deep Convolutional Neural Networks for Hyperspectral Image Classification,” J. Sensors, vol. 2015, p. 12, 2015.

**3D-CNN**: Y. Chen, H. Jiang, C. Li, X. Jia, and P. Ghamisi, “Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks,” IEEE Trans. Geosci. Remote Sens., vol. 54, pp. 6232–6251, 2016.

**A2S2K-ResNet**: S. K. Roy, S. Manna, T. Song, and L. Bruzzone, “Attention-Based Adaptive Spectral–Spatial Kernel ResNet for Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 9, pp. 7831–7843, 2021.

**FreeNet**: Z. Zheng, Y. Zhong, A. Ma, and L. Zhang, “FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 8, pp. 5612–5626, 2020.
