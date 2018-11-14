import torch
import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from cont_seg_with_semantic_net import Net
from train_cont_depth import Dfd, DfdConfig, RES_OUT


if __name__ == '__main__':
    torch.cuda.empty_cache()
    np.set_printoptions(linewidth=320)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DfdConfig(image_size='big', batch_size=1, mode='segmentation', target_mode='cont',lr=0.0001, get_reduced_dataset=False, num_classes=16,dataset='ours', end_epoch=300)
    net = Net(config=config, device=device, num_class=config.num_classes,mode=config.mode, channels=64,skip_layer=True)
    # net = net.to(device)
    # net = ResNet18(mode='segmentation', num_classes=16,image_size='small')
    dfd = Dfd(config=config, net=net, device=device)
    # dfd.resume(resume_path=os.path.join(RES_OUT,'discrete_segmentation_80t1_97pm1.pt'))
    dfd.resume()
    # dfd.resume(resume_path=os.path.join(RES_OUT, 'cont_segmentation_96pm1.pt'))
    # dfd.resume(resume_path=os.path.join(RES_OUT, 'best_model_raw_80top1_96pm1.pt'))
    # dfd.resume(resume_path=os.path.join(RES_OUT,'cont_from_scratch_59top1_94pm1.pt'), ordinal=True)
    # dfd.resume(resume_path=os.path.join(RES_OUT,'best_model.pt'))
    # dfd.show_imgs_and_seg(image_file='/home/yotamg/data/no_filter/real_image/out_15.raw',
    #                           label_file='/home/yotamg/data/depth_pngs/City_0212_rot1.png', patch_size=256)
    # dfd.show_imgs_and_seg(image_file='/home/yotamg/test/chk.raw',
    #                       label_file='/home/yotamg/data/depth_pngs/City_0212_rot1.png', patch_size=256)

    num_of_checkpoints = len(os.listdir('resources_out'))
    with open('/home/yotamg/data/no_filter/real_image/out_15.raw', 'rb') as f:
        img = pickle.load(f)
    f.close()
    # with open('/home/yotamg/test/chk.raw', 'rb') as f:
    #     img = pickle.load(f)
    # f.close()
    from PIL import Image
    # img = Image.fromarray(img)
    # img = img.resize((1000,1000))
    # img = np.array(img)
    # plt.subplot(4,4, 1)
    # plt.imshow(img)
    # plt.subplot(4,4, 2)
    # plt.imshow(img2)
    # img = np.array(img)
    img = dfd.prepare_for_net(img)
    # img2 = np.array(img2)
    # img2 = dfd.prepare_for_net(img2)

    # for i, checkpoint in enumerate(os.listdir('resources_out')):
    #     # print (checkpoint)
    #     dfd.resume(resume_path=os.path.join(RES_OUT,checkpoint))
        # dfd.resume()
        # dfd.show_imgs_and_seg()
    time_before = time.time()
    with torch.no_grad():
    #     for param in dfd.net.segmentation_module.parameters():
    #         param.requires_grad = False
        img1_predict = dfd.net(img)
    print ("Time for inference: ", time.time() - time_before)
        # img2_predict = dfd.net(img2)
        # plt.subplot(2,4,i+1)
        # plt.title(checkpoint)
        # plt.imshow(img1_predict, cmap='jet')
        # # plt.subplot(4,4,i+10)
        # # plt.title(checkpoint)
        # # plt.imshow(img2_predict, cmap='jet')
        # # break
    if dfd.config.target_mode == 'discrete':
        img1_predict = torch.squeeze(torch.argmax(img1_predict, dim=1),0)
    #     img2_predict = torch.squeeze(torch.argmax(img2_predict, dim=1),0)
    # print (img1_predict.shape)

    plt.figure(1)
    plt.imshow(img1_predict, cmap='jet')
    # plt.figure(2)
    # plt.imshow(img2_predict, cmap='jet')
    plt.show()
    print ("Done!")