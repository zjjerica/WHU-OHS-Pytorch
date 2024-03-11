# Training the network in "pixel-to-pixel" manner on public hyperspectral dataset

import torch
import math
import os
from CNN_3D import CNN_3D
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
import scipy.io as sio
import numpy as np

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

def main():
    model_name = '3DCNN'
    dataset_name = 'PU'
    class_num = 9
    epoch_num = 100

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

    s_file = sio.loadmat('./dataset_example_PU/PaviaU_gt_train.mat')
    gt = s_file['train']

    image = torch.tensor(image, dtype=torch.float)
    image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)

    gt = torch.tensor(gt, dtype=torch.long) - 1
    gt = gt.unsqueeze(0).to(device)

    print('Build model ...')
    model = CNN_3D(input_features=bands, n_classes=class_num).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    finetune = True
    if finetune:
        # Load the model pretrained on WHU-OHS dataset (for transfer learning)
        pretrained_path = '3DCNN_WHU_OHS_final.pth'
        pretrained_dict = torch.load(pretrained_path)
        del pretrained_dict['final_conv.weight']
        del pretrained_dict['final_conv.bias']
        model.load_state_dict(pretrained_dict, strict=False)
        model_path = './model/' + dataset_name + '_' + model_name + '_v2_finetune/'
        print('Loaded pretrained model.')
    else:
        model_path = './model/' + dataset_name + '_' + model_name + '_v2/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print('Start training.')

    for epoch in range(epoch_num):
        model.train()

        optimizer.zero_grad()

        pred = model(image)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()

        print('Epoch: %d training loss %.6f' % (epoch, loss.item()))

        # Save model regularly
        if (epoch % 5 == 0):
            torch.save(model.state_dict(), model_path + model_name + '_' + str(epoch) + '.pth')

    # Save model for the final epoch
    torch.save(model.state_dict(), model_path + model_name + '_final.pth')

if __name__ == '__main__':
    main()



