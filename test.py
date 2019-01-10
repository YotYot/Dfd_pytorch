from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
import argparse
from PIL import Image

# ours
from models import Net
from dataloader import Dataset
from skimage.measure import compare_ssim,compare_psnr
from utils import *
import torchvision.transforms as transforms


LR0 = 1e-7

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DfdNet Training')
    parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='load trained model')
    parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True, help='train model')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    parser.add_argument('--experiment_name', type=str, default='default', help='exp name')
    parser.add_argument('--target', type=str, default='segmentation', help='target - classificaion or segmentation')
    parser.add_argument('--target_mode', type=str, default='discrete', help='target mode - cont or discrete')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args)

    # set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/Filtered/rgb'
    # label_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/Filtered/GT'
    #
    # test_filelist        = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if 'alley_1' in img]
    # test_labels_filelist = [img.replace(test_dir,label_dir).replace('_1100_maskImg.png', '_GT.dpt') for img in test_filelist]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # test_dataset = Dataset(image_filelist=test_filelist, label_filelist=test_labels_filelist, transforms=transform, pickle_name='test.pickle', train=False)
    #
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    train_dir = '/home/yotamg/data/raw_rgb_pngs'
    train_filelist = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    test_filelist = train_filelist
    label_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Depth_sintel_and_tau_from_shay/'
    train_label_filelist = [img.replace(train_dir, label_dir).replace('.png', '.dpt') for img in train_filelist]
    test_label_filelist = train_label_filelist

    train_dataset = Dataset(image_filelist=train_filelist, label_filelist=train_label_filelist, train=True,
                            pickle_name='train_old_imgs_discrete.pickle', transforms=transform,
                            target_mode=args.target_mode)
    test_dataset = Dataset(image_filelist=test_filelist, label_filelist=test_label_filelist, train=False,
                           pickle_name='test_old_imgs_discrete.pickle', transforms=transform,
                           target_mode=args.target_mode)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=1)



    net = Net(device=device, mode=args.target, target_mode=args.target_mode)
    net = net.eval()
    net = net.to(device)
    load_model(net,device, fullpath='/home/yotamg/PycharmProjects/dfd/trained_models/Net_default/checkpoint_31.pth.tar')

    for i, data in enumerate(test_data_loader):
        # x,x_paths, y, y_paths = data
        x, x_path, y, y_path = data
        x = x.to(device)
        with torch.no_grad():
            out = net(x)
        out = out.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        plt.figure(1)
        out = np.argmax(out, axis=1)
        # out = np.squeeze(out,0)
        out = (out - np.min(out)) / (np.max(out) - np.min(out))
        ax1 = plt.subplot(1,3,2)
        ax1.imshow(y[0])
        ax2 = plt.subplot(1,3,1, sharex=ax1, sharey=ax1)
        x = (x + 1) / 2
        ax2.imshow(np.transpose(x[0], (1, 2, 0)))
        ax3 = plt.subplot(1, 3, 3, sharex=ax1,sharey=ax1)
        ax3.imshow(out[0])
        # plt.suptitle(label, fontsize="large")
        plt.show()

if __name__ == '__main__':
    main()
