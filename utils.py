import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_model_name(model):
    return model.__class__.__name__


def model_file_name(epoch):
    return 'checkpoint_'+str(epoch)+'.pth.tar'


def save_model(model, epoch, experiment_name):
    model_name = get_model_name(model) + '_' +experiment_name
    save_dir = os.path.join('./trained_models', model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_dict = {'state_dict': model.state_dict()}
    filename = model_file_name(epoch)

    filename = os.path.join(save_dir, filename)
    torch.save(model_dict, filename)


def load_model(model, device, epoch=300, experiment_name='default',fullpath=None):
    print("loading checkpoint")
    if fullpath:
        model_path = fullpath
    else:
        model_name = get_model_name(model) + '_' +experiment_name
        filename = model_file_name(epoch)
        model_path = os.path.join('./trained_models', model_name, filename)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

def _show_examples(x,y,out,num_of_examples=None):
    num_of_frames = num_of_examples if num_of_examples else x.shape[0]
    x = np.transpose(x.detach().cpu().numpy(), (0,2,3,1))
    x = (x+1)/2
    y = y.detach().cpu().numpy()
    #If target is discrete
    if out.shape[1] == 16:
        out = torch.argmax(out,dim=1)
    out = out.detach().cpu().numpy()
    for i in range(num_of_frames):
        plt.subplot(num_of_frames,3,3*i+1)
        plt.imshow(x[i])
        plt.subplot(num_of_frames, 3, 3*i + 2)
        plt.imshow(y[i])
        plt.subplot(num_of_frames, 3, 3*i + 3)
        plt.imshow(out[i])
    plt.show()

def _conf_matrix(predictions, labels, mat=None, partial=True):
    if predictions.shape[1] == 16:
        pred = torch.argmax(predictions,dim=1)
    else:
        pred = torch.clamp(torch.round(predictions),-4,10).long()
        labels = ((labels * 16)/ 255).long()
    pred = pred.view(-1)
    labels = labels.view(-1)

    if torch.min(pred) < 0:
        pred += 4
        labels += 4

    if partial:
        indices = random.sample(range(0, len(pred)), 10000)
    else:
        indices = torch.arange(0,len(pred))
    for i in indices:
        mat[labels[i]][pred[i]] += 1
    return mat

def _acc(conf_mat):
    conf_mat_no_label_0 = conf_mat[1:,1:]
    acc = torch.sum(torch.diag(conf_mat_no_label_0)).float() / (torch.sum(conf_mat_no_label_0)).float()
    acc_top3 = acc + torch.sum(torch.diag(conf_mat_no_label_0[1:,:])).float() / torch.sum(conf_mat_no_label_0).float() + torch.sum(torch.diag(conf_mat_no_label_0[:,1:])).float() / torch.sum(conf_mat_no_label_0).float()
    return acc, acc_top3


def _plot_fig(train_loss, val_loss, title):
    plt.title(title)
    plt.xlabel('epoch num'), plt.ylabel('loss')
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    file_name = title + '.png'
    plt.savefig(file_name)
    plt.show()

def _get_class_weights(class_items):
    # class_items = np.bincount(np.concatenate(np.concatenate(train_labels)).astype(np.uint8), minlength=16)
    class_weights = 1 / (class_items + 1)
    norm_class_weights = class_weights * (1 / min(class_weights))
    norm_class_weights[0] = 0
    return norm_class_weights

def get_patches_filelist(train_dir, test_dir):
    train_filelist = [os.path.join(train_dir, img) for img in os.listdir(train_dir) if img.endswith('.png')]
    test_filelist  = [os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith('.png')]
    return train_filelist, test_filelist