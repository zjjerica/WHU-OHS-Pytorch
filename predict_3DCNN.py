import torch
import os
import numpy as np
from dataset import WHU_OHS_Dataset
from CNN_3D import CNN_3D
from torchsummary import summary
from osgeo import gdal
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def writeTiff(im_data, im_width, im_height, im_bands, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main():
    print('Build model ...')
    model_name = '3DCNN'
    model = CNN_3D(input_features=32, n_classes=24).to(device)

    summary(model, (1, 32, 512, 512))

    # Load model (model of final epoch or best model evaluated on the validation set)
    model_path = './model/S1_3DCNN/3DCNN_final.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Load data ...')
    data_root = 'G:/WHU_OHS/'
    image_prefix = 'S1'

    data_path_test_image = os.path.join(data_root, 'ts', 'image')

    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_test_image)):
        for fname in fnames:
            if is_image_file(fname):
                if ((image_prefix + '_') in fname):
                    image_path = os.path.join(data_path_test_image, fname)
                    label_path = image_path.replace('image', 'label')
                    assert os.path.exists(label_path)
                    assert os.path.exists(image_path)
                    test_image_list.append(image_path)
                    test_label_list.append(label_path)

    assert len(test_image_list) == len(test_label_list)

    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list, use_3D_input=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print('Predicting.')

    save_path = './result/' + image_prefix + '_' + model_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        model.eval()
        for data, _, name in tqdm(test_loader):
            data = data.to(device)
            pred = model(data)
            output = pred[0, :, :, :].argmax(axis=0)
            output = output.cpu().detach().numpy() + 1
            output = output.astype(np.uint8)
            writeTiff(output, 512, 512, 1, os.path.join(save_path, name[0]))

if __name__ == '__main__':
    main()




