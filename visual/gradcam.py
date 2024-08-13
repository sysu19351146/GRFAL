
import torch
import torch.nn.functional as F
import cv2
#from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
import pandas as pd
matplotlib.use('Agg')
def gradcam_visual(model,image_path,random_noise,y):
    cam_dict={}
    model=model.cuda()
    pil_img = PIL.Image.open(image_path)
    #normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    #normed_torch_img = normalizer(torch_img)
    resnet_model_dict = dict(type='resnet', arch=model, layer_name='layer1', input_size=(224, 224))
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAM2(resnet_model_dict, True)
    #cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

    mask, pre =resnet_gradcam(torch_img,torch.tensor([y]).cuda(),random_noise=random_noise)
    mask2 = resnet_gradcampp(torch_img, torch.tensor([y]).cuda(),random_noise=random_noise)
    print(pre)
    heatmap, result = visualize_cam(mask, torch_img)
    heatmap2, result2 = visualize_cam(mask2, torch_img)
    # heatmap=normalizer.undo(heatmap)
    # result=normalizer.undo(result)
    return torch_img.squeeze().cpu(),heatmap,result,heatmap2, result2
    # images=[]
    # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap,  result], 0))
    # images = make_grid(torch.cat(images, 0), nrow=1)
    # save_image(images,'test.png')
    # kk=1

def show_visual_model(model,args):
    model_name=['Test Image','BAT','WAT','FAT','CFA','GDRO']
    name_list = ['BAT','WAT','FAT','CFA','GDRO']
    # net_paths = [
    #     "/share_data/cap_udr_test/"+"125-CUB-BAT-pgd-reTrue-roFalse-0.00010.00196/"+"robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/"+"123-CUB-WAT-pgd-reTrue-roFalse-0.00010.00196/"+"robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/"+"113-CUB-FAT-pgd-reTrue-roFalse-0.00010.00196/"+"robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/"+"137-CUB-CFA-pgd-reTrue-roFalse-0.00010.00196/"+"robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/"+"90-CUB-trades-pgd-reTrue-roTrue-0.00010.00196/"+"robustnew10best_test_model.pth",
    #     ]

    # net_paths = [
    #     "/share_data/cap_udr_test/" + "233-Skin-BAT-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/" + "255-Skin-WAT-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/" + "232-Skin-FAT-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/" + "253-Skin-CFA-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    #     "/share_data/cap_udr_test/" + "227-Skin-trades-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    # ]

    net_paths = [
        "/share_data/cap_udr_test/" + "112-CelebA-BAT-pgd-reTrue-roFalse-0.00010.00196/" + "robustnew10best_test_model.pth",
        "/share_data/cap_udr_test/" + "111-CelebA-WAT-pgd-reTrue-roFalse-0.00010.00196/" + "robustnew10best_test_model.pth",
        "/share_data/cap_udr_test/" + "115-CelebA-FAT-pgd-reTrue-roFalse-0.00010.00196/" + "robustnew10best_test_model.pth",
        "/share_data/cap_udr_test/" + "110-CelebA-CFA-pgd-reTrue-roFalse-0.00010.00196/" + "robustnew10best_test_model.pth",
        "/share_data/cap_udr_test/" + "208-CelebA-trades-pgd-reTrue-roTrue-0.00010.00196/" + "robustnew10best_test_model.pth",
    ]

    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    args.device = torch.device(f"cuda:{gpu_list[0]}"
                               if torch.cuda.is_available() and args.cuda else "cpu")
    model_list=[]

    for i in range(len(net_paths)):
        net_path = net_paths[i]
        model = torch.load(net_path, map_location=torch.device('cpu'))
        model = model.to(args.device)
        model_list.append(model)

    #f1=pd.read_csv('cub/data/waterbird_complete95_forest2water2/metadata.csv')
    f1 = pd.read_csv('celebA/data/list_attr_celeba.csv')
    #f1 = pd.read_csv('/share_data/dataset/gdroskin/metadata4.csv')
    for kk in range(50):
    #     image_path_list=['008.Rhinoceros_Auklet/Rhinoceros_Auklet_0005_2111.jpg',
    # '009.Brewer_Blackbird/Brewer_Blackbird_0137_2680.jpg',
    # '009.Brewer_Blackbird/Brewer_Blackbird_0041_2653.jpg',
    # '007.Parakeet_Auklet/Parakeet_Auklet_0042_795961.jpg',
    # '008.Rhinoceros_Auklet/Rhinoceros_Auklet_0013_797537.jpg',
    # ]
        mm=kk+1
        # image_path_list = [f1.iloc[mm*5]['img_filename'],
        #                    f1.iloc[mm*5+1]['img_filename'],
        #                    f1.iloc[mm*5+2]['img_filename'],
        #                    f1.iloc[mm*5+3]['img_filename'],
        #                    f1.iloc[mm*5+4]['img_filename'],
        #                    ]
        image_path_list = [f1.iloc[mm * 5]['image_id'],
                       f1.iloc[mm * 5 + 1]['image_id'],
                       f1.iloc[mm * 5 + 2]['image_id'],
                       f1.iloc[mm * 5 + 3]['image_id'],
                       f1.iloc[mm * 5 + 4]['image_id'],
                       ]
        # label_list=[f1.iloc[mm*5]['y'],
        #                    f1.iloc[mm*5+1]['y'],
        #                    f1.iloc[mm*5+2]['y'],
        #                    f1.iloc[mm*5+3]['y'],
        #                    f1.iloc[mm*5+4]['y']]

        label_list = [(f1.iloc[mm * 5]['Blond_Hair']==1)*1,
                      (f1.iloc[mm * 5 + 1]['Blond_Hair']==1)*1,
                       (f1.iloc[mm * 5 + 2]['Blond_Hair']==1)*1,
                        (f1.iloc[mm * 5 + 3]['Blond_Hair']==1)*1,
                         (f1.iloc[mm * 5 + 4]['Blond_Hair']==1)*1]

        # label_list = [1,
        #               0,
        #               0,
        #               1,
        #               1]
        origin_list=[]
        heatmap_list=[]
        result_list=[]
        heatmap2_list = []
        result2_list = []
        #torch_img = torch.from_numpy(np.asarray(PIL.Image.open('cub/data/waterbird_complete95_forest2water2/'+image_path_list[0]))).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
        #torch_img = torch.from_numpy(np.asarray(PIL.Image.open(image_path_list[0]))).permute(2, 0,1).unsqueeze(0).float().div(255).cuda()
        torch_img = torch.from_numpy(np.asarray(PIL.Image.open('celebA/data/img_align_celeba/' + image_path_list[0]))).permute(2, 0,1).unsqueeze(0).float().div(255).cuda()
        torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
        random_noise = torch.FloatTensor(torch_img.shape).uniform_(-0.0001, 0.0001).cuda()
        for i in range(len(net_paths)):
            net_path = net_paths[i]
            # ckpt = torch.load(
            #     net_path,
            #     map_location='cpu')

            # model =torch.load(net_path, map_location=torch.device('cpu'))
            #
            #
            # model = model.to(args.device)
            model=model_list[i]
            torch.cuda.set_device(args.device)
            heatmap_list_ = []
            result_list_ = []
            heatmap2_list_ = []
            result2_list_ = []
            for j in range(len(image_path_list)):
                #origin,heat,result,heat2,result2=gradcam_visual(model,'cub/data/waterbird_complete95_forest2water2/'+image_path_list[j],random_noise,label_list[j])
                origin, heat, result, heat2, result2 = gradcam_visual(model,
                                                                      'celebA/data/img_align_celeba/' +
                                                                      image_path_list[j], random_noise, label_list[j])

                #origin, heat, result, heat2, result2 = gradcam_visual(model, image_path_list[j], random_noise,label_list[j])
                heatmap_list_.append(heat.permute(1, 2, 0).numpy())
                result_list_.append(result.permute(1, 2, 0).numpy())
                heatmap2_list_.append(heat2.permute(1, 2, 0).numpy())
                result2_list_.append(result2.permute(1, 2, 0).numpy())
                origin_list.append(origin.permute(1, 2, 0).numpy())
            heatmap_list.append(heatmap_list_)
            result_list.append(result_list_)
            heatmap2_list.append(heatmap2_list_)
            result2_list.append(result2_list_)
            print(model_name[i+1])



        # fig, axes = plt.subplots(5, 6, figsize=(18, 15))
        #
        #
        # # 设置每个子图
        # for i in range(len(image_path_list)):
        #     img = origin_list[i]
        #     axes[i, 0].imshow(img)
        #     axes[i, 0].axis('off')  # 隐藏坐标轴
        #     # axes[i, 0].set_title(image_path_list[i][-16:-4], fontsize=15)
        #     for j in range(len(net_paths)):
        #         img=result_list[j][i]
        #         axes[i, j+1].imshow(img)
        #         axes[i, j+1].axis('off')  # 隐藏坐标轴

        for i in range(len(image_path_list)):
            img = origin_list[i]
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('zenunic/'+model_name[0]+'/{}_{}.png'.format(i,mm))
            # axes[i, 0].set_title(image_path_list[i][-16:-4], fontsize=15)
            for j in range(len(net_paths)):
                img=result_list[j][i]
                plt.axis('off')
                plt.imshow(img)
                plt.savefig('zenunic/' + model_name[j+1] + '/{}_{}.png'.format(i,mm))




        # 添加模型和测试图片的标签
        # for ax, col in zip(axes[0], model_name):
        #     # if col=='Test Image':
        #     #     continue
        #     ax.set_title(col, fontsize=15)
        #
        # plt.savefig('zenuni/model_comparison{}.png'.format(mm), dpi=600)
        #
        # fig_heat, axes_heat = plt.subplots(5, 6, figsize=(18, 15))
        #
        # for i in range(len(image_path_list)):
        #     img = origin_list[i]
        #     axes_heat[i, 0].imshow(img)
        #     axes_heat[i, 0].axis('off')  # 隐藏坐标轴
        #     for j in range(len(net_paths)):
        #         img = heatmap_list[j][i]
        #         axes_heat[i, j + 1].imshow(img)
        #         axes_heat[i, j + 1].axis('off')  # 隐藏坐标轴
        #
        # for ax, col in zip(axes_heat[0], model_name):
        #     ax.set_title(col, fontsize=15)

        # for i in range(len(image_path_list)):
        #     for j in range(len(net_paths)):
        #         img=heatmap_list[j][i]
        #         plt.imshow(img)
        #         plt.savefig('zenuni/' + model_name[j+1] + '/heat{}.png'.format(mm), dpi=600)

        #plt.tight_layout()
        # plt.savefig('zenuni/model_comparison_heat{}.png'.format(mm), dpi=600)
        #
        # fig, axes = plt.subplots(5, 6, figsize=(18, 15))
        #
        # # 设置每个子图
        # for i in range(len(image_path_list)):
        #     img = origin_list[i]
        #     axes[i, 0].imshow(img)
        #     axes[i, 0].axis('off')  # 隐藏坐标轴
        #     # axes[i, 0].set_title(image_path_list[i][-16:-4], fontsize=15)
        #     for j in range(len(net_paths)):
        #         img = result2_list[j][i]
        #         axes[i, j + 1].imshow(img)
        #         axes[i, j + 1].axis('off')  # 隐藏坐标轴
        #
        # # 添加模型和测试图片的标签
        # for ax, col in zip(axes[0], model_name):
        #     # if col == 'Test Image':
        #     #     continue
        #     ax.set_title(col, fontsize=15)
        #
        # plt.savefig('zenunimart/model_comparison{}.png'.format(mm), dpi=600)
        #
        # fig_heat, axes_heat = plt.subplots(5, 6, figsize=(18, 15))
        #
        # for i in range(len(image_path_list)):
        #     img = origin_list[i]
        #     axes_heat[i, 0].imshow(img)
        #     axes_heat[i, 0].axis('off')  # 隐藏坐标轴
        #     for j in range(len(net_paths)):
        #         img = heatmap2_list[j][i]
        #         axes_heat[i, j + 1].imshow(img)
        #         axes_heat[i, j + 1].axis('off')  # 隐藏坐标轴
        #
        # for ax, col in zip(axes_heat[0], model_name):
        #     ax.set_title(col, fontsize=15)
        #
        # # plt.tight_layout()
        # plt.savefig('zenunimart/model_comparison_heat{}.png'.format(mm), dpi=600)

        # for i in range(len(image_path_list)):
        #     img = origin_list[i]
        #     plt.imshow(img)
        #     plt.savefig('zenuni/'+model_name[0]+'/'+image_path_list[i][-16:-4]+'.png', dpi=600)
        #     # axes[i, 0].set_title(image_path_list[i][-16:-4], fontsize=15)
        #     for j in range(len(net_paths)):
        #         img=result_list[j][i]
        #         plt.imshow(img)
        #         plt.savefig('zenuni/' + model_name[j+1] + '/' + image_path_list[i][-16:-4] + '.png', dpi=600)
        #
        # for i in range(len(image_path_list)):
        #     for j in range(len(net_paths)):
        #         img=heatmap_list[j][i]
        #         plt.imshow(img)
        #         plt.savefig('zenuni/' + model_name[j+1] + '/heat' + image_path_list[i][-16:-4] + '.png', dpi=600)















def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().cpu()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None


        target_layer = find_resnet_layer(self.model_arch, layer_name)


        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.ta=target_layer
        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                #print('saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input, class_idx=None, retain_graph=False,random_noise=None):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.eval()
        x=input
        x_adv = x.detach()
        random_init = 0.0001
        epsilon = 0.00196
        step_size = epsilon / 4
        num_steps=10
        x_adv = Variable(x.data, requires_grad=True)
        criterion_kl2 = nn.KLDivLoss(reduction='none')
        criterion_kl = nn.KLDivLoss(size_average=False)
        loss_trade = torch.nn.CrossEntropyLoss(reduction='none')
        for k in range(num_steps):
            x_adv.requires_grad_()
            output = self.model_arch(x_adv)
            self.model_arch.zero_grad()
            with torch.enable_grad():
                loss_adv = nn.KLDivLoss(size_average=False)(F.log_softmax(output, dim=1),
                                                                F.softmax(self.model_arch(x.detach().clone()), dim=1))

            loss_adv.backward()
            eta = step_size * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model_arch.zero_grad()
        self.model_arch.train()
        plt.imshow((((torch.abs(x_adv - x)).squeeze(0).transpose(0, 2).transpose(0,1).cpu().detach().numpy()) / 0.005 * 255.0).astype(np.int))
        plt.savefig('noise.png')
        #
        # # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)
        #
        # x_adv = Variable(torch.clamp(x_adv, -3, 3), requires_grad=False)
        # self.model_arch.zero_grad()
        outputs = self.model_arch(x)
        loss_nat = loss_trade(outputs, class_idx)
        loss_robust = (1.0 / 1) * criterion_kl(F.log_softmax(self.model_arch(x_adv), dim=1),
                                                              F.softmax(self.model_arch(x), dim=1))


        #random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).cuda()
        #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        # for _ in range(10):
        #     opt = torch.optim.SGD([X_pgd], lr=1e-3)
        #     opt.zero_grad()
        #
        #     with torch.enable_grad():
        #         loss = nn.CrossEntropyLoss()(self.model_arch(X_pgd), class_idx)
        #     loss.backward()
        #     eta = step_size * X_pgd.grad.data.sign()
        #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        #     eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
        #     X_pgd = Variable(x.data + eta, requires_grad=True)
        #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=True)
        #
        # loss_robust=nn.CrossEntropyLoss()(self.model_arch(X_pgd), class_idx)
        # logit = self.model_arch(input)
        # if class_idx is None:
        #     score = logit[:, logit.max(1)[-1]].squeeze()
        # else:
        #     score = logit[:, class_idx].squeeze()
        score=loss_robust

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, torch.argmax(outputs)

    def __call__(self, input, class_idx=None, retain_graph=False,random_noise=None):
        return self.forward(input, class_idx, retain_graph,random_noise)

class GradCAM2(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None


        target_layer = find_resnet_layer(self.model_arch, layer_name)


        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.ta=target_layer
        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                #print('saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input, class_idx=None, retain_graph=False,random_noise=None):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        self.model_arch.eval()
        # x=input
        # x_adv = x.detach()
        # random_init = 0.0001
        # epsilon = 0.00196
        # step_size = epsilon / 4
        # num_steps=10
        # x_adv = Variable(x.data, requires_grad=True)
        # criterion_kl2 = nn.KLDivLoss(reduction='none')
        # criterion_kl = nn.KLDivLoss(size_average=False)
        # loss_trade = torch.nn.CrossEntropyLoss(reduction='none')
        # for k in range(num_steps):
        #     x_adv.requires_grad_()
        #     output = self.model_arch(x_adv)
        #     self.model_arch.zero_grad()
        #     with torch.enable_grad():
        #         loss_adv = nn.KLDivLoss(size_average=False)(F.log_softmax(output, dim=1),
        #                                                         F.softmax(self.model_arch(x.detach().clone()), dim=1))
        #
        #     loss_adv.backward()
        #     eta = step_size * x_adv.grad.sign()
        #     x_adv = x_adv.detach() + eta
        #     x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        #     x_adv = torch.clamp(x_adv, 0.0, 1.0)
        #
        # self.model_arch.zero_grad()
        # self.model_arch.train()
        #
        # # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)
        #
        # x_adv = Variable(torch.clamp(x_adv, -3, 3), requires_grad=False)
        # self.model_arch.zero_grad()
        # outputs = self.model_arch(x)
        # loss_nat = loss_trade(outputs, class_idx)
        # loss_robust = (1.0 / 1) * criterion_kl(F.log_softmax(self.model_arch(x_adv), dim=1),
        #                                                       F.softmax(self.model_arch(x), dim=1))


        #random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).cuda()
        #X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        # for _ in range(10):
        #     opt = torch.optim.SGD([X_pgd], lr=1e-3)
        #     opt.zero_grad()
        #
        #     with torch.enable_grad():
        #         loss = nn.CrossEntropyLoss()(self.model_arch(X_pgd), class_idx)
        #     loss.backward()
        #     eta = step_size * X_pgd.grad.data.sign()
        #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        #     eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
        #     X_pgd = Variable(x.data + eta, requires_grad=True)
        #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=True)
        #
        # loss_robust=nn.CrossEntropyLoss()(self.model_arch(X_pgd), class_idx)
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        # score=loss_robust

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False,random_noise=None):
        return self.forward(input, class_idx, retain_graph,random_noise)

class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit
    def __call__(self, input, class_idx=None, retain_graph=False,random_noise=None):
        return self.forward(input, class_idx, retain_graph)

def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)