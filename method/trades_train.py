import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
criterion_kl2 = nn.KLDivLoss(reduction='none')
criterion_kl = nn.KLDivLoss(size_average=False)
loss_trade = nn.CrossEntropyLoss(reduction='none')

def trades_train(model,x,y,g,is_training,loss_computer,args,optimizer):
    x_adv = trades(model, x, y, epsilon=args.epsilon, step_size=args.epsilon / 4,
                        num_steps=args.attack_iters,
                        loss_fn='trades', category='trades', rand_init=True, device=args.device)
    # model.eval()
    #
    # x_adv = x.detach() + args.random_init * torch.randn(x.shape).to(args.device).detach()
    #
    # for _ in range(args.attack_iters):
    #     output = model(x_adv)
    #     x_adv.requires_grad_()
    #     if args.early_stop:
    #         index = torch.where(output.max(1)[1] == y)[0]
    #     else:
    #         index = slice(None, None, None)
    #
    #     with torch.enable_grad():
    #         loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                F.softmax(model(x), dim=1))
    #         # loss_kl = loss_computer.loss_adv(output, y, g, is_training,
    #         #                                trades_attack=(args.beta * torch.sum(loss_kl, dim=1)))
    #     grad = torch.autograd.grad(loss_kl, [x_adv])[0]
    #     x_v = x_adv[index, :, :, :]
    #     g_v = grad[index, :, :, :]
    #     x_o = x[index, :, :, :]
    #     x_v = x_v.detach() - args.epsilon / args.attack_iters * torch.sign(g_v.detach())
    #     x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
    #     x_v = torch.clamp(x_v, args.clmin, args.clmax)
    #     x_adv.data[index, :, :, :] = x_v
    #     # x_adv.grad.zero_()

    model.train()

    # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

    # x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
    optimizer.zero_grad()
    outputs = model(x)
    loss_nat = loss_trade(outputs, y)
    # loss_robust =(1.0 / args.batch_size) *criterion_kl2(F.log_softmax(model(x_adv), dim=1),
    #                                                       F.softmax(model(x), dim=1))
    loss_robust = criterion_kl2(F.log_softmax(model(x_adv), dim=1),
                                F.softmax(model(x), dim=1))

    loss_main = loss_computer.loss(outputs, y, g, is_training,
                                   trade_loss=(loss_nat, args.beta * loss_robust.sum(dim=1)))

    optimizer.zero_grad()
    loss_main.backward()
    optimizer.step()


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