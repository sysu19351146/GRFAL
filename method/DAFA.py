import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def DAFA(model, x_natural, y,g, optimizer, args, class_weights, batch_indices=None, memory_dict=None):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""

    if args is not None:
        step_size, perturb_steps, epsilon = args.epsilon/4, args.attack_iters, args.epsilon
        beta = args.beta

    loss_dict = {}

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')

    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)

    ## dafa masking
    class_weights_mask = torch.zeros(len(y)).to(args.device)
    for i in range(args.n_classes):
        cur_indices = np.where(g.detach().cpu().numpy() == i)[0]
        class_weights_mask[cur_indices] = class_weights[i]

    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    x_adv += 0.0001 * torch.randn(x_natural.shape).to(args.device).detach()

    logits_nat = model(x_natural)

    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(logits, dim=1), F.softmax(logits_nat, dim=1))

        grad = torch.autograd.grad(loss_kl, [x_adv])[0]

        x_adv = x_adv.detach() + (class_weights_mask * step_size).view(-1, 1, 1, 1) * torch.sign(grad.detach())
        # class weights applied
        # class weights = 1.0 during warm-up and calculated values after the warm-up
        x_adv = torch.min(torch.max(x_adv, x_natural - (class_weights_mask * epsilon).view(-1, 1, 1, 1)),
                          x_natural + (class_weights_mask * epsilon).view(-1, 1, 1, 1))

        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(x_adv, requires_grad=False)

    model.train()
    optimizer.zero_grad()

    logits = model(x_adv)
    logits_nat = model(x_natural)

    # class weights applied
    loss_natural = (torch.nn.CrossEntropyLoss(reduction='none')(logits_nat, y) * class_weights_mask).mean()
    loss_dict['natural'] = loss_natural.item()

    p_natural = F.softmax(logits_nat, dim=1)
    loss_robust = criterion_kl(F.log_softmax(logits, dim=1), p_natural) / batch_size

    loss_dict['robust'] = loss_robust.item()
    loss = loss_natural + beta * loss_robust

    # save the statistics for calculating class weights, at the end of the warmup epoch

    if memory_dict is not None:
        # memory_dict['probs'][batch_indices.cpu().numpy()] = F.softmax(logits, dim=1).detach().cpu().numpy()
        # memory_dict['labels'][batch_indices.cpu().numpy()] = g.detach().cpu().numpy()

        memory_dict['probs'].append(F.softmax(logits, dim=1).detach().cpu().numpy())
        memory_dict['labels'].append(g.detach().cpu().numpy())


    return loss, loss_dict










def calculate_class_weights(memory_dict, lamb=1.0):
    probs, labels = memory_dict['probs'], memory_dict['labels']
    probs=np.vstack(probs)
    labels=np.hstack(labels)
    num_classes = probs.shape[1]
    class_similarity = np.zeros((num_classes, num_classes))
    class_weights = np.ones(num_classes)

    for i in range(num_classes):
        cur_indices = np.where(labels == i)[0]
        class_similarity[i] = np.mean(probs[cur_indices], axis=0)

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j: continue
            if class_similarity[i, i] < class_similarity[j, j]:
                class_weights[i] += lamb * class_similarity[i, j] * class_similarity[j, j]
            else:
                class_weights[i] -= lamb * class_similarity[i, j] * class_similarity[j, j]

    return class_weights