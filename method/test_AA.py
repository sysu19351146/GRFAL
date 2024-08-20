from autoattack import AutoAttack
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def AA_test(model,x,y,g,is_training,loss_computer,args,epsilon,random_init):
    model.eval()
    step_size = epsilon / 4
    adversary = AutoAttack(model, norm='Linf', eps=epsilon,
                           version='standard', device=args.device, verbose=False)
    adversary.attacks_to_run = ['apgd-ce']

    adv_images = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    adv_images.to(args.device)

    outputs = model(adv_images)
    loss_main = loss_computer.loss(outputs, y, g, is_training)