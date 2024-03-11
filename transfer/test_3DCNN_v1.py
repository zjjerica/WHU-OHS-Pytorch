# Testing the network using patch-based input on public hyperspectral dataset

import torch
import os
import numpy as np
from dataset_hyper import Hyper_Dataset_patch
from CNN_3D_patch import CNN_3D
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
    label = numClass * imgLabel + imgPredict
    count = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix

def main():
    model_name = '3DCNN'
    dataset_name = 'PU'
    class_num = 9
    batch_size = 100

    print('Load data ...')
    s_file = sio.loadmat('./dataset_example_PU/PaviaU.mat')
    image = s_file['paviaU']

    s_file = sio.loadmat('./dataset_example_PU/PaviaU_gt_test.mat')
    gt = s_file['test']

    bands = image.shape[2]

    print('Build model ...')
    model = CNN_3D(input_features=bands, n_classes=class_num).to(device)

    test_dataset = Hyper_Dataset_patch(image=image, gt=gt, windowsize=15, use_3D_input=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                               pin_memory=True, drop_last=False)

    # Load model
    model_path = './model/PU_3DCNN_finetune/3DCNN_final.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Testing.')

    with torch.no_grad():
        model.eval()
        confusionmat = torch.zeros([class_num, class_num])
        confusionmat = confusionmat.to(device)

        for data, label in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)

            output = pred.argmax(axis=1)

            confusionmat_tmp = genConfusionMatrix(class_num, output, label)
            confusionmat = confusionmat + confusionmat_tmp

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