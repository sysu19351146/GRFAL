import sys
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def write_dict_to_tb(writer, dict, prefix, step):
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def get_results(acc_groups, get_yp_func):
    groups = acc_groups.keys()
    results = {
            f"accuracy_{get_yp_func(g)[0]}_{get_yp_func(g)[1]}": acc_groups[g].avg
            for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results


def evaluate(model, loader, get_yp_func, multitask=False, predict_place=False,device=None):
    model.eval()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    if multitask:
        acc_place_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

    with torch.no_grad():
        for x, y, g, p in tqdm.tqdm(loader):
            x, y, p = x.to(device), y.to(device), p.to(device)
            if predict_place:
                y = p

            logits = model(x)
            if multitask:
                logits, logits_place = logits
                update_dict(acc_place_groups, p, g, logits_place)

            update_dict(acc_groups, y, g, logits)
    model.train()
    if multitask:
        return get_results(acc_groups, get_yp_func), get_results(acc_place_groups, get_yp_func)
    return get_results(acc_groups, get_yp_func)

def evaluate_pgd(model, loader, get_yp_func, multitask=False, predict_place=False,device=None):
    model.eval()
    acc_groups = {g_idx: AverageMeter() for g_idx in range(loader.dataset.n_groups)}
    if multitask:
        acc_place_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

    with torch.no_grad():
        for x, y, g, p in tqdm.tqdm(loader):
            x, y, p = x.to(device), y.to(device), p.to(device)
            if predict_place:
                y = p
            x_adv=pgd_test(model,x,y,epsilon=0.00196,random_init=0.0001,device=device)
            logits = model(x_adv)
            if multitask:
                logits, logits_place = logits
                update_dict(acc_place_groups, p, g, logits_place)

            update_dict(acc_groups, y, g, logits)
    model.train()
    if multitask:
        return get_results(acc_groups, get_yp_func), get_results(acc_place_groups, get_yp_func)
    return get_results(acc_groups, get_yp_func)


class MultiTaskHead(nn.Module):
    def __init__(self, n_features, n_classes_list,device=None):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [
            nn.Linear(n_features, n_classes).to(device)
            for n_classes in n_classes_list
        ]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from autoattack import AutoAttack

def pgd_test(model,x,y,epsilon=0.00196,random_init=0.0001,device=None):
    model.eval()
    step_size = epsilon / 4
    X_pgd = Variable(x.data, requires_grad=True)

    random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(10):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
        X_pgd = Variable(x.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=True)
    x = X_pgd
    return x



def AA_test(model,x,y,epsilon=0.00196,random_init=0.0001,device=None):
    model.eval()
    step_size = epsilon / 4

    adversary = AutoAttack(model, norm='Linf', eps=epsilon,
                           version='standard', device=device, verbose=False)
    adversary.attacks_to_run = ['apgd-ce']

    adv_images = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    adv_images.to(device)
    return x


criterion_kl2 = nn.KLDivLoss(reduction='none')
criterion_kl = nn.KLDivLoss(size_average=False)
loss_trade = nn.CrossEntropyLoss()
def trades_train(model,x,y,optimizer,device):
    x_adv = trades(model, x, y, epsilon=0.00196, step_size=0.00196/ 4,
                        num_steps=10,
                        loss_fn='trades', category='trades', rand_init=True, device=device)


    model.train()

    # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

    # x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
    optimizer.zero_grad()
    outputs = model(x)
    loss_nat = loss_trade(outputs, y)
    loss_robust =(1.0 / x.shape[0]) *criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                          F.softmax(model(x), dim=1))

    loss=loss_nat+6* loss_robust
    return loss





def trades(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,device):
    model.eval()
    if category == "trades":
        x_adv = data.detach().clone() + 0.0001 * torch.randn(data.shape).to(device) if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            elif loss_fn == 'trades':
                loss_adv = nn.KLDivLoss(size_average=False)(F.log_softmax(output, dim=1),
                                       F.softmax(model(data.detach().clone()), dim=1))

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.zero_grad()
    return x_adv.detach()