# Training the network using patch-based input on public hyperspectral dataset

import torch
import os
from dataset_hyper import Hyper_Dataset_patch
from CNN_3D_patch import CNN_3D
import torch.optim as optim
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

def main():
    model_name = '3DCNN'
    dataset_name = 'PU'
    class_num = 9
    epoch_num = 50
    batch_size = 100

    print('Load data ...')
    s_file = sio.loadmat('./dataset_example_PU/PaviaU.mat')
    image = s_file['paviaU']

    s_file = sio.loadmat('./dataset_example_PU/PaviaU_gt_train.mat')
    gt = s_file['train']

    bands = image.shape[2]

    print('Build model ...')
    model = CNN_3D(input_features=bands, n_classes=class_num).to(device)

    finetune = True
    if finetune:
        # Load the model pretrained on WHU-OHS dataset (for transfer learning)
        pretrained_path = '3DCNN_WHU_OHS_final.pth'
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        model_path = './model/' + dataset_name + '_' + model_name + '_finetune/'
        print('Loaded pretrained model.')
    else:
        model_path = './model/' + dataset_name + '_' + model_name + '/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataset = Hyper_Dataset_patch(image=image, gt=gt, windowsize=15, use_3D_input=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    print('Start training.')

    for epoch in range(epoch_num):
        print('Epoch: %d/%d' % (epoch + 1, epoch_num))
        print('Current learning rate: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']))

        model.train()
        batch_index = 0
        loss_sum = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.item()
            batch_index = batch_index + 1
            average_loss_cur = loss_sum / batch_index
            if (batch_index % 5 == 0):
                print('training loss %.6f' % (average_loss_cur))

        average_loss = loss_sum / batch_index
        print('Epoch [%d/%d] training loss %.6f' % (epoch + 1, epoch_num, average_loss))

        # Save model regularly
        if (epoch % 5 == 0):
            torch.save(model.state_dict(), model_path + model_name + '_' + str(epoch) + '.pth')

    # Save model for the final epoch
    torch.save(model.state_dict(), model_path + model_name + '_final.pth')

if __name__ == '__main__':
    main()



