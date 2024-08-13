import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
def CFA(model, x_natural, y, g, cw_eps, cw_beta, step_size, perturb_steps):
    batch_beta = cw_beta[g]
    batch_eps = cw_eps[g]
    base = torch.ones_like(x_natural).to(x_natural.device)
    for sample in range(len(x_natural)):
        base[sample] *= batch_eps[sample]
    batch_eps = base.clone().to(x_natural.device)

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.0001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - batch_eps), x_natural + batch_eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y, reduction='none') / batch_size
    cw_criterion = nn.KLDivLoss(reduction='none')
    robust_out = model(x_adv)
    loss_robust = (1.0 / batch_size) * cw_criterion(F.log_softmax(robust_out, dim=1),
                                                    F.softmax(model(x_natural), dim=1))

    assert len(batch_beta) == len(loss_robust)
    loss_robust = loss_robust.sum(1)
    #print(batch_beta.shape, loss_natural.shape, loss_robust.shape)
    loss = ((1-batch_beta) * loss_natural + 6*batch_beta * loss_robust).sum()

    return loss, robust_out.clone().detach()


class CW_log():
    def __init__(self,device, class_num=10) -> None:
        self.N = 0
        self.robust_acc = 0
        self.clean_acc = 0
        self.cw_robust = torch.zeros(class_num).to(device)
        self.cw_clean = torch.zeros(class_num).to(device)
        self.class_num = class_num
        self.eps=8/255
        self.beta=6
        self.alpha=2/255
        self.class_eps=torch.ones(class_num)
        self.class_beta = torch.ones(class_num)

    def update_clean(self, output, y,g):
        self.N += len(output)
        pred = output.max(1)[1]
        correct = pred == y
        self.clean_acc += correct.sum()

        for i, c in enumerate(g):
            if correct[i]:
                self.cw_clean[c] += 1

    def update_robust(self, output, y,g):
        pred = output.max(1)[1]
        correct = pred == y
        self.robust_acc += correct.sum()

        for i, c in enumerate(g):
            if correct[i]:
                self.cw_robust[c] += 1

    def result(self):
        N = self.N
        m = self.class_num
        return self.clean_acc / N, self.robust_acc / N, m * self.cw_clean / N, m * self.cw_robust / N


def weight_average(model, new_model, decay_rate, init=False):
    model.eval()
    new_model.eval()
    state_dict = model.state_dict()
    new_dict = new_model.state_dict()
    if init:
        decay_rate = 0
    for key in state_dict:
        new_dict[key] = (state_dict[key]*decay_rate + new_dict[key]*(1-decay_rate)).clone().detach()
    model.load_state_dict(new_dict)