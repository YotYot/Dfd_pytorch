import os
import h5py
import numpy as np
from sklearn.utils import shuffle
from collections import namedtuple
import tqdm
import torchfile
import pickle
import random
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
from os import path
from warnings import warn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sintel_io import depth_read


TrainPickle = 'train.pickle'
TestPickle  = 'test.pickle'

labels_histo_pickle = 'labels_histo.pickle'

pickle_dir = './pickles'

def image_read_and_extend(image_path):
    image_arr = Image.open(image_path)
    w, h = image_arr.size
    if h%32 != 0:
        new_size = (1024, 512)
        new_im = Image.new("RGB", new_size)
        new_im.paste(image_arr, (0,(new_size[1] - h) // 2))
        image_arr = new_im
    return image_arr

def depth_read_and_extend(depth_path):
    depth = depth_read(depth_path)
    if depth.shape[0] % 32 != 0:
        depth = np.pad(depth, ((38,38),(0,0)), 'constant')
    if np.min(depth) < 0:
        depth += 5
    return depth

def patch_read(patch_path):
    label = int(patch_path.split("_")[-1].split('.')[0])
    return label

class Dataset(data.Dataset):
    def __init__(self, image_filelist, label_filelist, image_loader=image_read_and_extend, label_loader=depth_read_and_extend, train=True, pickle_name=None, transforms=None, target_mode='discrete'):
        self.image_filelist = image_filelist
        self.label_filelist = label_filelist
        self.train = train
        self.image_loader = image_loader
        self.label_loader = label_loader
        self.transforms = transforms
        self.target_mode = target_mode
        if pickle_name:
            pickle_filename = pickle_name
        else:
            pickle_filename = TrainPickle if train else TestPickle
        pickle_path = os.path.join(pickle_dir, pickle_filename)
        if not os.path.isfile(pickle_path):
            self.generate_pickle(self.image_filelist, self.label_filelist,pickle_path)
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def generate_pickle(self, image_filelist, label_filelist, pickle_path):
        imgs = list()
        imgs_path = list()
        labels = list()
        lbls_path = list()
        for image_path in tqdm.tqdm(image_filelist):
            try:
                image_arr = self.image_loader(image_path)
            except:
                print ("Skipping " + image_path)
                continue
            imgs.append(image_arr)
            imgs_path.append(image_path)

        for label_path in tqdm.tqdm(label_filelist):
            try:
                label_arr = self.label_loader(label_path)
            except:
                print("Skipping " + label_path)
                continue
            labels.append(label_arr)
            lbls_path.append(label_path)

        with open(pickle_path, 'wb') as f:
            pickle.dump([imgs, imgs_path, labels, lbls_path], f)

    def get_labels_hisotgram(self):
        pickle_path = os.path.join(pickle_dir, labels_histo_pickle)
        if not os.path.isfile(pickle_path):
            histo = np.bincount(np.concatenate(self.dataset[2]).ravel().astype(np.uint8), minlength=16)
            with open(pickle_path, 'wb') as f:
                pickle.dump(histo, f)
        with open(pickle_path, 'rb') as f:
            histo = pickle.load(f)
        return histo

    def noisy(self,image, sigma=3):
        row, col, ch = image.shape
        mean = 0
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        return gauss.astype(np.int8)

    def __getitem__(self, index):
        imgs,imgs_paths,label, label_paths = self.dataset[0][index], self.dataset[1][index],self.dataset[2][index], self.dataset[3][index]

        hflip = random.random() < 0.5
        if hflip:
            imgs = imgs.transpose(Image.FLIP_LEFT_RIGHT)
            #If its classification, don't change label
            if not isinstance(label, int):
                label = np.flip(label,1)

        vflip = random.random() < 0.5
        if vflip:
            imgs = imgs.transpose(Image.FLIP_TOP_BOTTOM)
            # If its classification, don't change label
            if not isinstance(label, int):
                label = np.flip(label,0)

        imgs = np.array(imgs).astype(np.int16)
        imgs = np.clip(imgs + self.noisy(imgs),0,255)
        imgs = imgs.astype(np.uint8)
        imgs = Image.fromarray(imgs)

        imgs = self.transforms(imgs)

        if self.target_mode == 'discrete':
            label = label.astype(np.long)

        if not isinstance(label, int):
            label = label.copy()

        return imgs,imgs_paths,label, label_paths


    def __len__(self):
        return len(self.dataset[0])

    def name(self):
        return 'Dataset'



def example():
    train_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/left_filtered_adapted_all_but_alley_1'
    test_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/left_filtered_adapted_alley_1'
    label_dir = '/media/yotamg/bd0eccc9-4cd5-414c-b764-c5a7890f9785/Yotam/Sintel/depth_flatten'
    train_files = os.listdir(train_dir)
    train_filelist = [os.path.join(train_dir, img) for img in train_files]
    label_filelist = [img.replace(train_dir, label_dir).replace('.png','.dpt') for img in train_filelist]

    dataset = Dataset(train_filelist, label_filelist, train=True)
    print('Dataset:', dataset)


if __name__ == '__main__':
    example()
