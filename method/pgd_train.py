import torch
import torch.nn.functional as F
from torch.autograd import Variable
def pgd_train(model,x,y,g,is_training,loss_computer,args,optimizer):
    model.eval()
    step_size=args.epsilon/4
    x_adv = x.detach() + args.random_init * torch.randn(x.shape).to(args.device).detach()
    for _ in range(args.attack_iters):
        opt = torch.optim.SGD([ x_adv], lr=1e-3)
        opt.zero_grad()
        x_adv.requires_grad_()
        output = model(x_adv)
        # if args.early_stop:
        #     index = torch.where(output.max(1)[1] == y)[0]
        # else:
        #     index = slice(None, None, None)
        # with torch.enable_grad():
        #     loss_ce = F.cross_entropy(output, y)
        # grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        # x_v = x_adv[index, :, :, :]
        # g_v = grad[index, :, :, :]
        # x_o = x[index, :, :, :]
        # x_v = x_v.detach() + args.epsilon / 4 * torch.sign(g_v.detach())
        # x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
        # x_v = torch.clamp(x_v, args.clmin, args.clmax)
        # x_adv.data[index, :, :, :] = x_v
        with torch.enable_grad():
            loss_ce = F.cross_entropy(output, y)

        loss_ce.backward()
        eta = step_size *  x_adv.grad.data.sign()
        x_adv = Variable( x_adv.data + eta, requires_grad=True)
        eta = torch.clamp( x_adv.data - x.data, -args.epsilon, args.epsilon)
        x_adv = Variable(x.data + eta, requires_grad=True)
        x_adv = Variable(torch.clamp( x_adv, args.clmin, args.clmax), requires_grad=True)
    x= Variable(torch.clamp( x_adv, args.clmin, args.clmax), requires_grad=False)
    model.zero_grad()
    optimizer.zero_grad()
    model.train()
    outputs = model(x)
    # loss_main = loss_computer.loss(outputs, y, g, is_training,trade_loss=(loss_trade(outputs, y),loss_trade(model(x_adv.detach()), y)))
    loss_main = loss_computer.loss(outputs, y, g, is_training)
    optimizer.zero_grad()
    loss_main.backward()
    optimizer.step()
