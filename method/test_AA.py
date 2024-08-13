from autoattack import AutoAttack
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def AA_test(model,x,y,g,is_training,loss_computer,args,epsilon,random_init):
    model.eval()
    step_size = epsilon / 4
    # X_pgd = Variable(x.data, requires_grad=True)
    #
    # random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).to(args.device)
    # X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    #
    # for _ in range(10):
    #     opt = torch.optim.SGD([X_pgd], lr=1e-3)
    #     opt.zero_grad()
    #
    #     with torch.enable_grad():
    #         loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    #     loss.backward()
    #     eta = step_size * X_pgd.grad.data.sign()
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    #     eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
    #     X_pgd = Variable(x.data + eta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, args.clmin, args.clmax), requires_grad=True)
    # x = X_pgd
    adversary = AutoAttack(model, norm='Linf', eps=epsilon,
                           version='standard', device=args.device, verbose=False)
    adversary.attacks_to_run = ['apgd-ce']

    adv_images = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    adv_images.to(args.device)

    outputs = model(adv_images)
    loss_main = loss_computer.loss(outputs, y, g, is_training)