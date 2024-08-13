import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





def WAT(model,x,y,g,is_training,loss_computer,args,optimizer):




    output_adv = trades(model, x, y, epsilon=args.epsilon, step_size=args.epsilon/4,
                               num_steps=args.attack_iters,
                               loss_fn='trades', category='trades', rand_init=True,device=args.device)
    # set model.train after be attacked
    model.train()
    optimizer.zero_grad()

    natural_logits = model(x)
    adv_logits = model(output_adv)

    iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits, natural_logits, y,device=args.device)

    for i in range(args.class_num):
        if i == 0:
            nat_loss = iter_nat_loss[g == i].sum() * args.nat_class_weights[i]
            bndy_loss = iter_bndy_loss[g == i].sum() * args.bndy_class_weights[i]
        else:
            nat_loss += iter_nat_loss[g == i].sum() * args.nat_class_weights[i]
            bndy_loss += iter_bndy_loss[g== i].sum() * args.bndy_class_weights[i]
    loss = (nat_loss + args.beta * bndy_loss) / x.shape[0]

    # In WAT, we add excepted training loss in the decision set.

    loss += TRADES_loss(adv_logits, natural_logits, y, args.beta,device=args.device) * args.bndy_class_weights[args.class_num]


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # model.eval()
    # outputs = model(x)
    # loss_main = loss_computer.loss(outputs, y, g, is_training)


def validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost,args,epoch,device):
    val_iter = 0
    model.eval()
    epoch_val_bndy_loss = torch.zeros(args.class_num + 1).to(device)
    epoch_val_bndy_loss.requires_grad = False
    epoch_val_nat_loss = torch.zeros(args.class_num + 1).to(device)
    epoch_val_nat_loss.requires_grad = False
    correct = 0
    val_class_wise_acc = []
    val_class_wise_num = []
    for i in range(args.class_num):
        val_class_wise_acc.append(0)
        val_class_wise_num.append(0)
    model.zero_grad()
    for batch_idx, batch in enumerate(valid_loader):
        batch = tuple(t.to(args.device) for t in batch)
        data = batch[0]
        target = batch[1]
        group = batch[2]
        val_iter += 1


        output_adv = trades(model, data, target, args.epsilon, args.epsilon/4, args.attack_iters, "trades",
                                   "trades", True,device=device)

        with torch.no_grad():
            natural_logits = model(data)
            adv_logits = model(output_adv)
            iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits, natural_logits, target,device=device)
            pred = adv_logits.max(1, keepdim=True)[1]
            target_view = group.view_as(pred)
            eq_mask = pred.eq(target_view)  # equal mask
            # class-wise
            for i in range(args.class_num):
                label_mask = target_view == i
                val_class_wise_num[i] += label_mask.sum().item()
                val_class_wise_acc[i] += (eq_mask * label_mask).sum().item()
                epoch_val_nat_loss[i] += iter_nat_loss[group == i].sum()
                epoch_val_bndy_loss[i] += iter_bndy_loss[group == i].sum()
            epoch_val_nat_loss[args.class_num] += iter_nat_loss.sum() / args.class_num
            epoch_val_bndy_loss[args.class_num] += iter_bndy_loss.sum() / args.class_num

            correct += pred.eq(target.view_as(pred)).sum().item()

    wat_valid_nat_cost[epoch] = epoch_val_nat_loss / (val_iter * args.batch_size) * args.class_num
    wat_valid_bndy_cost[epoch] = epoch_val_bndy_loss / (val_iter * args.batch_size) * args.class_num

    val_acc = correct / len(valid_loader.dataset)
    for i in range(args.class_num):
        val_class_wise_acc[i] /= val_class_wise_num[i]
    model.zero_grad()
    return wat_valid_nat_cost, wat_valid_bndy_cost, val_acc, val_class_wise_acc




def TRADES_loss(adv_logits, natural_logits, target, beta,device):
    # Based on the repo of TREADES: https://github.com/yaodongyu/TRADES
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).to(device)
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def TRADES_classwise_loss(adv_logits, natural_logits, target,device):
    criterion_kl = nn.KLDivLoss(reduction='none').to(device)
    loss_natural = nn.CrossEntropyLoss(reduction='none')(natural_logits, target)
    loss_robust = criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    return loss_natural, loss_robust.sum(dim=1)

def trades(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,device):
    model.eval()
    if category == "trades":
        x_adv = data.detach().clone() + 0.0001* torch.randn(data.shape).to(device) if rand_init else data.detach()
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