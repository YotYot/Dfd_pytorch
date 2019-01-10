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
    parser.add_argument('--target_mode', type=str, default='cont', help='target mode - cont or discrete')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = Net(device=device, mode=args.target, target_mode=args.target_mode)
    net = net.eval()
    net = net.to(device)
    load_model(net,device, fullpath='trained_models/Net_continuous_trained/checkpoint_274.pth.tar')

    imgs_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Real-Images/png'
    imgs_filelist = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir) if img.endswith('.png')]

    for i, img in enumerate(imgs_filelist):
        # x,x_paths, y, y_paths = data
        x = plt.imread(img)
        x = np.expand_dims(x,0)
        x = np.transpose(x, (0,3,1,2))
        x = x[:,:,2:-2,8:-8]
        x = torch.Tensor(x).to(device)
        with torch.no_grad():
            out = net(x)
        out = out.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        plt.figure(1)
        if args.target_mode == 'discrete':
            out = np.argmax(out, axis=1)
            out = out[0]
        # out = np.squeeze(out,0)
        out = (out - np.min(out)) / (np.max(out) - np.min(out))
        ax1 = plt.subplot(1,3,1)
        # x = (x + 1) / 2
        ax1.imshow(np.transpose(x[0], (1, 2, 0)))
        ax3 = plt.subplot(1, 3, 3, sharex=ax1,sharey=ax1)
        ax3.imshow(out, cmap='jet')
        # plt.suptitle(label, fontsize="large")
        plt.show()

if __name__ == '__main__':
    main()
