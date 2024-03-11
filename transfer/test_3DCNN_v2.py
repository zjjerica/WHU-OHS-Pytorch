# Testing the network in "pixel-to-pixel" manner on public hyperspectral dataset

import torch
import os
import numpy as np
from CNN_3D import CNN_3D
from tqdm import tqdm
from torchsummary import summary
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def genConfusionMatrix(numClass, imgPredict, imgLabel):
    mask = (imgLabel != -1)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

def main():
    model_name = '3DCNN'
    dataset_name = 'PU'
    class_num = 9

    print('Load data ...')
    s_file = sio.loadmat('./dataset_example_PU/PaviaU.mat')
    image = s_file['paviaU']

    bands = image.shape[2]

    image = image.astype(np.float)
    for i in range(bands):
        temp = image[:, :, i]
        mean = np.mean(temp)
        std = np.std(temp)
        if (std == 0):
            image[:, :, i] = 0
        else:
            image[:, :, i] = (temp - mean) / std

    s_file = sio.loadmat('./dataset_example_PU/PaviaU_gt_test.mat')
    gt = s_file['test']

    image = torch.tensor(image, dtype=torch.float)
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

    gt = torch.tensor(gt, dtype=torch.long) - 1
    gt = gt.unsqueeze(0).to(device)

    print('Build model ...')
    model = CNN_3D(input_features=bands, n_classes=class_num).to(device)

    # Load model
    model_path = './model/PU_3DCNN_v2_finetune/3DCNN_final.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Testing.')

    with torch.no_grad():
        model.eval()
        pred = model(image)
        pred = pred[0, :, :, :].argmax(axis=0)
        gt = gt[0, :, :]
        confusionmat = genConfusionMatrix(class_num, pred, gt)

    confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 0.0000001

    PA = a / (c + eps)
    UA = a / (b + eps)
    print('PA:', PA)
    print('UA:', UA)

    F1 = 2 * PA * UA / (PA + UA + eps)
    print('F1:', F1)

    mean_F1 = np.nanmean(F1)
    print('mean F1:', mean_F1)

    OA = np.sum(a) / np.sum(confusionmat)
    print('OA:', OA)

    PE = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (OA - PE) / (1 - PE)
    print('Kappa:', Kappa)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    print('IoU:', IoU)
    print('mIoU:', mIoU)

if __name__ == '__main__':
    main()