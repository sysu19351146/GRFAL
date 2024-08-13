import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

criterion_kl = nn.KLDivLoss(reduction='none')
criterion_nat = nn.CrossEntropyLoss(reduction='none')




def FAT(model, data, y, g,args,optimizer):
    weight0, weight1, weight2 = match_weight(y, args.FAT_args.diff0, args.FAT_args.diff1, args.FAT_args.diff2,g,args.class_num,args.device)

    ## generate attack
    x_adv = trades_adv(model, data, weight2,args.device, args.attack_iters,args.epsilon/4,args.epsilon)

    model.train()
    ## get loss
    loss_natural = criterion_nat(model(data), y)
    loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))

    # loss_natural_avg = torch.sum(loss_natural * weight0) / torch.sum(weight0)
    loss_robust1 = torch.sum(loss_robust, 1)
    loss = torch.sum(weight0 * loss_natural + weight1 * loss_robust1) / torch.sum(weight0 + weight1)

    ## back propagates
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


class FAT_args():
    def __init__(self) -> None:
        self.diff0=1
        self.diff1=1
        self.diff2=1


def test_valid_FAT(model, loader,args):

    print('Currently Testing the Unfairness on Training Set')
    all_label = []
    all_pred = []
    all_pred_adv = []
    all_g=[]
    model.eval()
    correct = 0
    correct_adv = 0

    for batch_idx, batch in enumerate(loader):
        batch = tuple(t.to(args.device) for t in batch)
        data = batch[0]
        target = batch[1]
        g = batch[2]


        all_label.append(target)
        all_g.append(g)

        ## clean test
        output = model(data)
        pred = output.max(dim=1)[1]
        all_pred.append(pred)
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add

        ## generate PGD attack
        adv_samples = pgd_attack(model, data, target,args.epsilon, args.clmax,args.clmin,args.attack_iters,args.epsilon/4)
        output = model(adv_samples)
        pred_adv = output.max(dim=1)[1]
        all_pred_adv.append(pred_adv)
        add1 = pred_adv.eq(target.view_as(pred_adv)).sum().item()
        correct_adv += add1


    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()
    all_g=torch.cat(all_g).flatten()

    acc = in_class(all_pred, all_label,all_g,args.class_num)
    acc_adv = in_class(all_pred_adv, all_label,all_g,args.class_num)

    total_clean_error = 1- correct / all_label.shape[0]
    total_bndy_error = correct / all_label.shape[0] - correct_adv / all_label.shape[0]

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error

def in_class(predict, label,group,num_class):

    probs = torch.zeros(num_class)
    for i in range(num_class):
        in_class_id = torch.tensor(group == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs


def match_weight(label, diff0, diff1, diff2,group,num_class,device):

    weight0 = torch.zeros(label.shape[0], device=device)
    weight1 = torch.zeros(label.shape[0], device=device)
    weight2 = torch.zeros(label.shape[0], device=device)

    for i in range(num_class):
        weight0 += diff0[i] * torch.tensor(group == i, dtype= torch.float).to(device)
        weight1 += diff1[i] * torch.tensor(group == i, dtype= torch.float).to(device)
        weight2 += diff2[i] * torch.tensor(group == i, dtype= torch.float).to(device)

    weight2 = torch.exp(2 * weight2)
    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2,class_num):

    diff0 = torch.zeros(class_num)
    for i in range(class_num):
        for j in range(class_num):
            if j == i:
                diff0[i] = diff0[i] + (class_num-1) / class_num * lam0[i]
            else:
                diff0[i] = diff0[i] - (class_num-1) / class_num * lam0[j]
        diff0[i] = diff0[i] + 1 / class_num

    diff1 = torch.zeros(class_num)
    for i in range(class_num):
        for j in range(class_num):
            if j == i:
                diff1[i] = diff1[i] + (class_num-1) / class_num * lam1[i]
            else:
                diff1[i] = diff1[i] - (class_num-1) / class_num * lam1[j]
        diff1[i] = diff1[i] + 1 / class_num

    diff2 = lam2

    diff0 = torch.clamp(diff0, min = 0)
    diff1 = torch.clamp(diff1, min = 0)

    return diff0, diff1, diff2

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size,
                  ):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)



        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()


    return X_pgd

def trades_adv(model,
               x_natural,
               weight2,
               device,
               num_steps= 6,
               step_size= 3 / 255,
               epsilon= 12 /255,
               distance='l_inf'
               ):

    new_eps = (epsilon * weight2).view(weight2.shape[0], 1, 1, 1)

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.0001 * torch.randn(x_natural.shape).to(device).detach()

    if distance == 'l_inf':
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            eta = step_size * torch.sign(grad.detach())
            eta = torch.min(torch.max(eta, -1.0 * new_eps), new_eps)
            x_adv = x_adv.detach() + eta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv