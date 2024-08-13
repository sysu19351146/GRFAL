import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import random
import numpy as np

def BAT(model,x,y,g,is_training,loss_computer,args,optimizer):
    model.train()
    last_clean, lastclean_target, _, _ = stop_to_lastclean(model, x, y, False,
                                                           step_size=args.epsilon/4,
                                                           epsilon=args.epsilon,
                                                           perturb_steps=args.attack_iters,
                                                           randominit_type="normal_distribution_randominit",
                                                           loss_fn='kl',
                                                           device=args.device)


    first_adv, _, output_natural, _ = stop_to_firstadv(model, x, y, step_size=args.epsilon/4,
                                                       epsilon=args.epsilon, perturb_steps=args.attack_iters,
                                                       randominit_type="normal_distribution_randominit",
                                                       loss_fn='kl', tau=1,
                                                       device=args.device)

    model.train()
    optimizer.zero_grad()
    lastclean_logits = model(last_clean)
    firstadv_logits = model(first_adv)
    clean_logits = model(output_natural)

    loss_natural, loss_robust, loss_uniform = BAT_loss(firstadv_logits, lastclean_logits, lastclean_target,
                                                       args.beta, clean_logits,device=args.device)
    loss = loss_robust + loss_natural + loss_uniform

    loss.backward()
    optimizer.step()

    # model.eval()
    # outputs = model(x)
    # loss_main = loss_computer.loss(outputs, y, g, is_training)





def BAT_loss(firstadv_logits, lastclean_logits, target, beta, clean_logits,device):
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).to(device)
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(lastclean_logits, target)
    uniform_logits = torch.ones_like(firstadv_logits).to(device) * 0.1
    loss_uniform_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(uniform_logits, dim=1),
                                                         F.softmax(lastclean_logits, dim=1))
    loss_uniform_robust2 = (1.0 / batch_size) * criterion_kl(F.log_softmax(uniform_logits, dim=1),
                                                         F.softmax(firstadv_logits, dim=1))
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(firstadv_logits, dim=1),
                                                         F.softmax(clean_logits, dim=1))
    loss_uniform = (loss_uniform_robust + loss_uniform_robust2)
    return loss_natural, beta * loss_robust, loss_uniform




def stop_to_firstadv(model, data, target, step_size, epsilon, perturb_steps,randominit_type,device,loss_fn='kl',tau=1,rand_init=True,omega=0,img_size=224):
    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []
    index_list = []
    index = [num for num in range(len(data))]
    random.shuffle(index)
    img_size = data.shape[2]
    control = (torch.ones(len(target)) * tau).to(device)

    # Initialize the adversarial data with random noise
    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device)
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    else:
        iter_adv = data.to(device).detach()

    iter_clean_data = data.to(device).detach()
    iter_index_flag = torch.tensor([num for num in range(len(iter_clean_data))])
    iter_target = target.to(device).detach()
    output_iter_clean_data = model(data)

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, img_size, img_size).to(device)
                output_natural = iter_clean_data[output_index].reshape(-1, 3, img_size, img_size).to(device)
                output_target = iter_target[output_index].reshape(-1).to(device)
                output_index_flag = iter_index_flag[output_index].reshape(-1)
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, img_size, img_size).to(device)), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, img_size, img_size).to(device)), dim=0)
                output_index_flag = torch.cat((output_index_flag, iter_index_flag[output_index].reshape(-1)), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).to(device)), dim=0)

        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).to(device)
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_index_flag = iter_index_flag[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().to(device)
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv[index], output_target[index], output_natural[index], output_index_flag
            # if random input
            # return output_adv[index], output_target, output_natural, output_index_flag
        K = K-1

    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().to(device)
        output_adv = iter_adv.reshape(-1, 3, img_size, img_size).to(device)
        output_natural = iter_clean_data.reshape(-1, 3, img_size, img_size).to(device)
        output_index_flag = iter_index_flag.reshape(-1).squeeze()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, img_size, img_size)), dim=0).to(device)
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().to(device)
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, img_size, img_size).to(device)),dim=0).to(device)
        output_index_flag = torch.cat((output_index_flag, iter_index_flag.reshape(-1)), dim=0)
    output_adv = output_adv.detach()
    return output_adv[index], output_target[index], output_natural[index], output_index_flag
    # if random input
    # return output_adv[index], output_target, output_natural, output_index_flag


def stop_to_lastclean(model, data, target, print_flag, step_size, epsilon, perturb_steps,randominit_type,loss_fn,device,tau=10,rand_init=True,omega=0,img_size=224):
    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []

    control = (torch.ones(len(target))).to(device)
    img_size=data.shape[2]
    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device)
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    else:
        iter_adv = data.to(device).detach()

    iter_clean_data = data.to(device).detach()
    iter_index_flag = torch.tensor([num for num in range(len(iter_clean_data))])
    iter_target = target.to(device).detach()
    output_iter_clean_data = model(data)

    tmp_adv = iter_adv

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                output_index.append(idx)
            else:
                iter_index.append(idx)

        if len(output_index) != 0:
            if len(output_target) == 0:
                output_adv = tmp_adv[output_index].reshape(-1, 3, img_size, img_size).to(device)
                output_natural = iter_clean_data[output_index].reshape(-1, 3, img_size, img_size).to(device)
                output_target = iter_target[output_index].reshape(-1).to(device)
                output_index_flag = iter_index_flag[output_index].reshape(-1)
            else:
                output_adv = torch.cat((output_adv, tmp_adv[output_index].reshape(-1, 3, img_size, img_size).to(device)), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, img_size, img_size).to(device)), dim=0)
                output_index_flag = torch.cat((output_index_flag, iter_index_flag[output_index].reshape(-1)), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).to(device)), dim=0)


        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).to(device)
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            tmp_adv = iter_adv
            iter_clean_data = iter_clean_data[iter_index]
            iter_index_flag = iter_index_flag[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().to(device)
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)

            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_target, output_natural, output_index_flag
        K = K-1



    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().to(device)
        output_adv = iter_adv.reshape(-1, 3, img_size, img_size).to(device)
        output_natural = iter_clean_data.reshape(-1, 3, img_size, img_size).to(device)
        output_index_flag = iter_index_flag.reshape(-1).squeeze()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, img_size, img_size)), dim=0).to(device)
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().to(device)
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, img_size, img_size).to(device)),dim=0).to(device)
        output_index_flag = torch.cat((output_index_flag, iter_index_flag.reshape(-1)), dim=0)
    output_adv = output_adv.detach()

    return output_adv, output_target, output_natural, output_index_flag