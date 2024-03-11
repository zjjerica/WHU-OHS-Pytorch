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

The correspondence of label IDs and categories:

<div align="center">

<table>
    <tr>
        <th align="center">ID
        <th align="center">Category
        <th align="center">ID
        <th align="center">Category
        <th align="center">ID
        <th align="center">Category
        <th align="center">ID
        <th align="center">Category
    </tr>
    <tr>
        <td align="left">1
        <td align="left">Paddy field
        <td align="left">7
        <td align="left">High-covered grassland
        <td align="left">13
        <td align="left">Beach land
        <td align="left">19
        <td align="left">Gobi
    </tr>
    <tr>
        <td align="left">2
        <td align="left">Dry farm
        <td align="left">8
        <td align="left">Medium-covered grassland
        <td align="left">14
        <td align="left">Shoal
        <td align="left">20
        <td align="left">Saline-alkali soil
    </tr>
    <tr>
        <td align="left">3
        <td align="left">Woodland
        <td align="left">9
        <td align="left">Low-covered grassland
        <td align="left">15
        <td align="left">Urban built-up
        <td align="left">21
        <td align="left">Marshland
    </tr>
    <tr>
        <td align="left">4
        <td align="left">Shrubbery
        <td align="left">10
        <td align="left">River canal
        <td align="left">16
        <td align="left">Rural settlement
        <td align="left">22
        <td align="left">Bare land
    </tr>
    <tr>
        <td align="left">5
        <td align="left">Sparse woodland
        <td align="left">11
        <td align="left">Lake
        <td align="left">17
        <td align="left">Other construction land
        <td align="left">23
        <td align="left">Bare rock
    </tr>
    <tr>
        <td align="left">6
        <td align="left">Other forest land
        <td align="left">12
        <td align="left">Reservoir pond
        <td align="left">18
        <td align="left">Sand
        <td align="left">24
        <td align="left">Ocean
    </tr>

</table>

</div>


## Code
Pytorch toolbox for large-scale hyperspectral image classification using WHU-OHS dataset. The deep network models will be updated continuously.

**Update:**

**2024.3.11 Example code for transfer learning from WHU-OHS dataset to public hyperspectral datasets is updated. See the "transfer" folder.**

Deep network models:

**1D-CNN**: W. Hu, Y. Huang, L. Wei, F. Zhang, and H. Li, “Deep Convolutional Neural Networks for Hyperspectral Image Classification,” J. Sensors, vol. 2015, p. 12, 2015.

**3D-CNN**: Y. Chen, H. Jiang, C. Li, X. Jia, and P. Ghamisi, “Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks,” IEEE Trans. Geosci. Remote Sens., vol. 54, pp. 6232–6251, 2016.

**A2S2K-ResNet**: S. K. Roy, S. Manna, T. Song, and L. Bruzzone, “Attention-Based Adaptive Spectral–Spatial Kernel ResNet for Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sens., vol. 59, no. 9, pp. 7831–7843, 2021.

**FreeNet**: Z. Zheng, Y. Zhong, A. Ma, and L. Zhang, “FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 8, pp. 5612–5626, 2020.

Accuracy on test set for reference (taking S1: Changchun as an example):

<div align="center">

<table>
    <tr>
        <th align="center">Network
        <td align="center">1D-CNN
        <td align="center">3D-CNN
        <td align="center">A2S2K-ResNet
        <td align="center">FreeNet
    </tr>
    <tr>
        <th align="center">OA
        <td align="center">0.636
        <td align="center">0.766
        <td align="center">0.809
        <td align="center">0.847
    </tr>
    <tr>
        <th align="center">Kappa
        <td align="center">0.526
        <td align="center">0.700
        <td align="center">0.757
        <td align="center">0.806
    </tr>
    <tr>
        <th align="center">mIoU
        <td align="center">0.227
        <td align="center">0.305
        <td align="center">0.419
        <td align="center">0.480
    </tr>
</table>

</div>
