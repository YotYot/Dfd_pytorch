import torch
import torchvision
import torchvision.transforms as transforms
from depth_classification_dataset import depth_classification_dataset
from depth_segmentation_dataset import depth_segmentation_dataset
from resnet import ResNet18, ResNet50
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


RES_OUT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources_out')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 9)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 5)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, 5)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 15, 1)
        self.dense = nn.Linear(20*20*32, 15)


    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.dropout2d(x)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)
        x = F.dropout2d(x)
        x = F.relu(self.conv5(x))
        x = x.view(-1, 15)
        return x

def load_data(mode, image_size, batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'segmentation':
        if image_size == 'small':
            train_dir = '/home/yotamg/data/jpg_images/patches'
            test_dir = '/home/yotamg/data/jpg_images/test/patches'
            label_dir = '/home/yotamg/data/depth_pngs/patches'
        else:
            train_dir = '/home/yotamg/data/jpg_images/'
            test_dir = '/home/yotamg/data/jpg_images/test/'
            label_dir = '/home/yotamg/data/depth_pngs/'
    else:
        train_dir = '/home/yotamg/data/rgb/train'
        test_dir = '/home/yotamg/data/rgb/val'
        label_dir = None

    if mode == 'segmentation':
        trainset = depth_segmentation_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                              label_dir=label_dir, load_pickle=True)
        testset = depth_segmentation_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                             label_dir=label_dir, load_pickle=True)
    else:
        trainset = depth_classification_dataset(root='./data', train=True, transform=transform, train_dir=train_dir,
                                                load_pickle=True)
        testset = depth_classification_dataset(root='./data', train=False, transform=transform, test_dir=test_dir,
                                               load_pickle=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=2)

    return trainloader, testloader


class DfdConfig(object):
    def __init__(self, batch_size=64, mode='segmentation', image_size='small', resume_path=os.path.join(RES_OUT,'best_model.pt'), start_epoch=0, end_epoch=50, model_state_name='best_model.pt', num_classes=15, lr=0.01):
        self.batch_size = batch_size
        self.mode = mode
        self.image_size = image_size
        self.resume_path = resume_path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.model_state_name = model_state_name
        self.num_classes = num_classes
        self.lr = lr

class Dfd(object):
    def __init__(self, net=None, config=DfdConfig(), device=None):
        self.config = config
        self.trainloader, self.testloader = load_data(config.mode,config.image_size, config.batch_size)
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net if net else ResNet18(mode=config.mode, image_size=config.image_size)
        self.net = (self.net).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14')
        self.colors = (
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128))
        self.loss_history = list()
        self.acc_history = list()
        self.acc_plus_minus_history = list()
        self.steps_history = list()

    # functions to show an image
    def prepare_for_net(self, np_arr):
        np_arr = np.expand_dims(np_arr, 0)
        np_arr = np.transpose(np_arr, (0, 3, 1, 2))
        np_arr = torch.from_numpy(np_arr)
        np_arr = np_arr.type(torch.FloatTensor)
        np_arr = np_arr.to(self.device)
        return np_arr

    def imshow(self, img, transpose=True):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if transpose:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        else:
            plt.imshow(npimg)

    def plot_metrics(self):
        plt.plot(self.steps_history, self.loss_history, color='g')
        plt.plot(self.steps_history, self.acc_history, color='b')
        plt.plot(self.steps_history, self.acc_plus_minus_history, color='r')
        plt.xlabel('Steps')
        plt.ylabel('loss (g), acc (b), acc+-1 (r)')
        plt.show()


    def show_random_training_images(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        if not self.config.mode == 'segmentation':
            print(' '.join('%5s' % self.classes[labels[j]] for j in range(self.config.batch_size)))


    def resume(self):
        if self.config.resume_path:
            if os.path.isfile(self.config.resume_path):
                print("=> loading checkpoint '{}'".format(self.config.resume_path))
                checkpoint = torch.load(self.config.resume_path)
                self.start_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.config.resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.config.resume_path))


    def show_imgs_and_seg(self, image_file=None, label_file=None):
        self.net = self.net.eval()
        with torch.no_grad():
            if not image_file:
                for data in self.testloader:
                    images, labels = data
                    outputs = self.net(images.to(self.device))
                    predicted = torch.argmax(outputs.data, dim=1)
                    for i in range(4):
                        plt.subplot(2, 4, i)
                        self.imshow(images[i])
                        plt.subplot(2, 4, i+4)
                        plt.imshow(predicted[i])
                        plt.show()
            else:
                img = plt.imread(image_file)
                img_x = img.shape[0]
                img_y = img.shape[1]
                end_x = img_x // 32
                end_y = img_y // 32
                img = torch.from_numpy(img)
                img_patches_predictions = torch.zeros((end_x, end_y,32,32))
                img_predictions = np.zeros((img_x, img_y), dtype=np.uint8)
                img_predictions_image = np.zeros((img_predictions.shape[0],img_predictions.shape[1], 3), dtype=np.uint8)
                big_image = Image.open(image_file)
                big_image = np.array(big_image)
                big_image = self.prepare_for_net(big_image)
                big_image_pred = self.net(big_image)
                big_image_predicted = torch.argmax(big_image_pred, dim=1)
                big_image_predicted = big_image_predicted.cpu().numpy()
                big_image_predicted = np.squeeze(big_image_predicted, axis=0)
                for i in range(end_x):
                    for j in range(end_y):
                        patch = img[i * 32:(i + 1) * 32:, j * 32:(j + 1) * 32:, :]
                        patch = self.prepare_for_net(patch)
                        outputs = self.net(patch)
                        predicted = torch.argmax(outputs.data, dim=1)
                        img_patches_predictions[i, j, : , :] = predicted
                for i in range(end_x):
                    for j in range(end_y):
                        img_predictions[i*32:(i+1)*32, j*32:(j+1)*32] = (img_patches_predictions[i,j]).numpy()
                labels = plt.imread(label_file)
                labels_img = np.zeros((labels.shape[0], labels.shape[1],3), dtype=np.uint8)
                labels = (((labels - np.min(labels)) / (np.max(labels) - np.min(labels))) * 14).astype(np.uint8)
                for i in range(15):
                    img_predictions_image[img_predictions == i,:] = self.colors[i]
                    labels_img[labels == i,:] = self.colors[i]
                plt.subplot(4,1,1)
                self.imshow(img, transpose=False)
                plt.subplot(4,1,2)
                plt.imshow(labels_img)
                plt.subplot(4,1,3)
                plt.imshow(img_predictions_image)
                plt.subplot(4,1,4)
                plt.imshow(big_image_predicted)
                plt.show()


    def evaluate_model(self, partial=False):
        self.net = self.net.eval()
        correct = 0
        total = 0
        correct_plus_minus = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                if self.config.mode == 'segmentation':
                    total += labels.flatten().size()[0]
                    predicted = torch.argmax(outputs.data, dim=1)
                    labels = labels.long()
                    correct += (predicted == labels).sum().item()
                    labels_plus = labels + 1
                    labels_minus = labels - 1
                    correct_exact = (predicted == labels)
                    correct_plus  = (predicted == labels_plus)
                    correct_minus = (predicted == labels_minus)
                    correct_plus_minus += (correct_exact | correct_plus | correct_minus).sum().item()
                    if partial and total >= 100000: #About 1000 images
                        break
                else:
                    total += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    if partial and total >= 10000:
                        break
        acc = 100 * correct / total
        acc_plus_minus = 100*correct_plus_minus / total
        print('Num of values compared: ', total)
        print('Accuracy on val images: %d %%' % (acc))
        print('Accuracy on label or -/+ 1 of the label: %d %%' % (acc_plus_minus))
        return acc, acc_plus_minus


    def train(self):
        acc = 1
        best_acc = 0
        self.evaluate_model(partial=True)
        for epoch in range(self.config.start_epoch, self.config.end_epoch):  # loop over the dataset multiple times
            self.net = self.net.train()
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % (8000/self.config.batch_size) == (8000/self.config.batch_size)-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (8000/self.config.batch_size)))
                    self.loss_history.append(running_loss)
                    self.steps_history.append(i+1)
                    running_loss = 0.0

            prev_acc = acc
            acc, acc_plus_minus = self.evaluate_model(partial=True)
            self.acc_history.append(acc)
            self.acc_plus_minus_history.append(acc_plus_minus)
            if acc > best_acc:
                best_acc = acc
                print ("Saving model with best accuracy so far...")
                state = {
                    'epoch': epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                if not os.path.isdir(RES_OUT):
                    os.mkdir(RES_OUT)
                torch.save(state, os.path.join(RES_OUT, self.config.model_state_name))
            if (acc < prev_acc):
                current_lr = self.optimizer.defaults['lr']
                if current_lr > 1e-6:
                    print ("Reducing learning rate from " + str(current_lr) +" to ", str(current_lr/10))
                    self.optimizer = optim.Adam(self.net.parameters(), lr=current_lr/10)
                    # self.optimizer = optim.SGD(self.net.parameters(), lr=current_lr/10, momentum=0.9)
        print('Finished Training')

    def acc_per_class(self):
        class_correct = list(0. for i in range(self.config.num_classes))
        class_total = list(0. for i in range(self.config.num_classes))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(self.config.num_classes):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    config = DfdConfig(image_size='big')
    dfd = Dfd(config=config)
    # dfd.resume()
    dfd.train()
    dfd.plot_metrics()
    dfd.show_imgs_and_seg(image_file='/home/yotamg/data/jpg_images/City_0212_rot1.jpg', label_file='/home/yotamg/data/depth_pngs/City_0212_rot1.png')