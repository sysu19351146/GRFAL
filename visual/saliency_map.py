import os

import cv2
import importlib
import numpy as np
from PIL import Image
from captum.attr import Saliency, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import torch.nn as nn
from src.visualize.utils.image import *
#preprocess_image = importlib.import_module(f"src.visualize.utils.image").preprocess_image
from torch.autograd import Variable
import shutil
preprocess_image = importlib.import_module(f"src.visualize.utils.image").preprocess_image
show_cam_on_image = importlib.import_module(f"src.visualize.utils.image").show_cam_on_image
cam_method = importlib.import_module(f"src.visualize.transAttention_cam").TransformerCam

def generate(model, opt, prefix='clean',method='trades'):
    #image_folder=os.getcwd()
    image_folder='outputs/image'
    image_folder=os.listdir(image_folder)
    i=0
    for image_item in image_folder :
        if image_item[-3:]=='png' :

            model.eval()
            model.zero_grad()

            saliency = Saliency(model)

            nt = NoiseTunnel(saliency)

            rgb_img = Image.open(os.path.join('outputs/image',image_item)).convert('RGB')

            input_tensor = preprocess_image(rgb_img).to(opt.device)
            target_category = opt.image_label


            # saliency_map = saliency.attribute(input_tensor, target=target_category, abs=True)

            # stdevs = (torch.max(input_tensor).item() - torch.min(input_tensor).item()) * 0.02
            # print(stdevs)
            stdevs = 0.01
            saliency_map = nt.attribute(inputs=input_tensor, nt_type='smoothgrad', nt_samples=50, target=target_category,
                                        stdevs=stdevs)

            fig, _ = viz.visualize_image_attr(np.transpose(saliency_map.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              outlier_perc=opt.outlier_perc, use_pyplot=False)
            if not os.path.exists(opt.folder_path + "/" +image_item[:-4]):
                os.mkdir(opt.folder_path + "/" +image_item[:-4])
            fig.savefig(opt.folder_path + "/" +image_item[:-4]+'/'+ prefix + method + opt.visualize_path, bbox_inches='tight', pad_inches = -0.1, dpi=100)

            input_tensor, _, _ = pgd_whitebox_train(model, input_tensor,
                                                    torch.tensor(target_category).unsqueeze(0).to(opt.device), 0.02,
                                                    step_size=0.02 / 10)

            #plt.subplot(1, 1, 1)
            plt.imshow((((input_tensor).squeeze(0).transpose(0, 2).transpose(0,1).cpu().detach().numpy()) / 1 * 255.0).astype(
                np.int))
            plt.axis('off')
            plt.savefig(opt.folder_path + "/" +image_item[:-4]+'/'+ prefix + method + 'advpictrue', bbox_inches='tight', pad_inches=-0.1,
                        dpi=100)

            # saliency_map = saliency.attribute(input_tensor, target=target_category, abs=True)

            # stdevs = (torch.max(input_tensor).item() - torch.min(input_tensor).item()) * 0.02
            # print(stdevs)
            stdevs = 0.01
            saliency_map = nt.attribute(inputs=input_tensor, nt_type='smoothgrad', nt_samples=50, target=target_category,
                                        stdevs=stdevs)

            fig, _ = viz.visualize_image_attr(np.transpose(saliency_map.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(input_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              outlier_perc=opt.outlier_perc, use_pyplot=False)

            fig.savefig(opt.folder_path + "/" +image_item[:-4]+'/'+ prefix + method + 'adv'+opt.visualize_path, bbox_inches='tight', pad_inches=-0.1,
                        dpi=100)
            rgb_img=rgb_img.resize((442,442),resample=Image.BILINEAR)
            rgb_img.save(os.path.join(opt.folder_path + "/" +image_item[:-4],image_item))
            #rgb_img.show()
            #shutil.copy(os.path.join('outputs/image',image_item),os.path.join(opt.folder_path + "/" +image_item[:-4],image_item))
            print(i)
            i+=1

    # x = Image.open(opt.folder_path + "/" + prefix + "-2" + opt.visualize_path+'.png').convert('RGB')
    # img_array=np.array(x)
    # for k in range(img_array.shape[0]):
    #     for j in range(img_array.shape[1]):
    #         if img_array[k][j][0]<=20 and img_array[k][j][1]<=20 and img_array[k][j][1]<=20:
    #             img_array[k][j][0] = 255
    #             img_array[k][j][1] = 255
    #             img_array[k][j][2] = 255
    # img2=Image.fromarray(img_array)
    # img2.save(opt.folder_path + "/" + prefix + "-3" + opt.visualize_path+'.png')
    # pass
    # fig=
    # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(opt.folder_path + "/"+prefix + "-"+opt.model_method +"-"+ opt.visual_method + "-" + opt.model_name
    #             + "-" + opt.visualize_path, cam_image)

def pgd_whitebox_train(model,
                  X,
                  y,
                  epsilon,
                  num_steps=10,
                  step_size=0.003):
    out = model(X)
    X_pgd = Variable(X.data, requires_grad=True)

    #random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    #y_=torch.tensor([3]).cuda()
    for _ in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            #loss = nn.CrossEntropyLoss()(model(X_pgd), y_.long())
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    pred=model(X_pgd).data.max(1)[1]
    err_pgd = (pred != y.data).float().sum()
    return X_pgd,err_pgd,pred