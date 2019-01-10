from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
from utils import _show_examples,_acc,_plot_fig,_conf_matrix,_get_class_weights, save_model,load_model, get_patches_filelist

# ours
from models import Net
from dataloader import Dataset, patch_read

LR0 = 1e-3

def train(net, train_data_loader, test_data_loader,device,num_epochs=10):
    args = get_args()
    net.to(device)
    batch_size = train_data_loader.batch_size
    if args.target_mode == 'cont':
        criterion = nn.MSELoss()
    else:
        class_weights = _get_class_weights(train_data_loader.dataset.get_labels_hisotgram())
        class_weights = torch.from_numpy(class_weights).float().to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=0, weight=class_weights)
        # criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(net.parameters(), lr=LR0)
    optimizer.zero_grad()

    net.zero_grad()
    train_epochs_loss = []
    val_epochs_loss = []
    train_steps_per_e = len(train_data_loader.dataset) // batch_size
    val_steps_per_e   = len(test_data_loader.dataset) // batch_size
    best_loss = 1e5
    best_acc = 0
    for e in range(num_epochs):
        print ("Epoch: ", e)
        net = net.train()
        val_loss_sum = 0.
        train_loss_sum = 0
        for i, data in enumerate(train_data_loader):
            x,_,y,_ = data
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            optimizer.zero_grad()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print('Step: {:3} / {:3} Train loss: {:3.3}'.format(i, train_steps_per_e,loss.item()))
            train_loss_sum += loss.item()
        train_epochs_loss += [train_loss_sum / train_steps_per_e]
        conf_mat = torch.zeros((16, 16), dtype=torch.long)
        net = net.eval()
        for i, val_data in enumerate(test_data_loader):
            x,_,y,_ = val_data
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out = net(x)
            loss = criterion(out, y)
            val_loss_sum += loss.item()
            conf_mat = _conf_matrix(out,y,conf_mat, partial=False)
        val_epochs_loss += [val_loss_sum / val_steps_per_e]
        acc, acc_top3 = _acc(conf_mat)
        if val_epochs_loss[-1] < best_loss or acc > best_acc:
            print ("Saving Model")
            save_model(net, epoch=e, experiment_name=get_args().experiment_name)
            print (conf_mat)
            # _show_examples(x,y,out)
            best_loss = val_epochs_loss[-1]
            best_acc = acc
        print("\nepoch {:3} Train Loss: {:1.5} Val loss: {:1.5} Acc: {:1.5} Top3 Acc: {:1.5}".format(e, train_epochs_loss[-1],
                                                                                                     val_epochs_loss[-1], acc, acc_top3))
    return train_epochs_loss, val_epochs_loss


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DepthNet Training')
    parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='load trained model')
    parser.add_argument('--load_path', type=str, default='/home/yotamg/PycharmProjects/dfd/trained_models/Net_default/checkpoint_41.pth.tar',
                        help='load trained model path')
    parser.add_argument('--train', type=lambda x: (str(x).lower() == 'true'), default=True, help='train model')
    parser.add_argument('--epochs', type=int, default=300, help='num of epochs')
    parser.add_argument('--experiment_name', type=str, default='patches', help='exp name')
    parser.add_argument('--target', type=str, default='classification', help='target - classificaion or segmentation')
    parser.add_argument('--target_mode', type=str, default='classification', help='target mode - cont, discrete or classification')
    args = parser.parse_args()
    return args

def main():
    torch.set_printoptions(linewidth=320)
    args = get_args()
    print(args)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patches_train_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Patches/train'
    patches_test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Patches/test'

    train_filelist, test_filelist = get_patches_filelist(patches_train_dir, patches_test_dir)
    train_label_filelist, test_label_filelist = train_filelist, test_filelist

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = Dataset(image_filelist=train_filelist, label_filelist=train_label_filelist, train=True,
                            pickle_name='train_patches.pickle', transforms=transform, target_mode=args.target_mode,
                            label_loader=patch_read)
    test_dataset = Dataset(image_filelist=test_filelist, label_filelist=test_label_filelist, train=False,
                           pickle_name='test_patches.pickle', transforms=transform, target_mode=args.target_mode,
                           label_loader=patch_read)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=1)

    net = Net(device=device, mode=args.target, target_mode=args.target_mode)
    model_name = 'DepthNet'

    if args.load_model:
        load_model(net, device,fullpath=args.load_path)

    if args.train:
        train_loss, val_loss = train(net=net, train_data_loader=train_data_loader, test_data_loader=test_data_loader, device=device, num_epochs=args.epochs)
        _plot_fig(train_loss, val_loss, model_name+'-Losses')
        save_model(net, epoch=args.epochs, experiment_name=args.experiment_name)

if __name__ == '__main__':
    main()
