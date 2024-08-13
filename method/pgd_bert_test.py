import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


def pgd_bert_test(model,surrogate_model,x,y,g,is_training,loss_computer,optimizer,attack_bert_test,epsilon,random_init):
    surrogate_model.train()
    surrogate_model.zero_grad()

    K = 10
    input_ids = x[:, :, 0]
    input_masks = x[:, :, 1]
    segment_ids = x[:, :, 2]
    outputs = surrogate_model(
        input_ids=input_ids,
        attention_mask=input_masks,
        token_type_ids=segment_ids,
        labels=y
    )[1]

    loss = F.cross_entropy(outputs, y)

    loss.backward()  # 反向传播，得到正常的grad
    attack_bert_test.backup_grad()
    # 对抗训练
    for t in range(K):
        attack_bert_test.attack(
            is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
        if t != K - 1:
            surrogate_model.zero_grad()
        else:
            attack_bert_test.restore_grad()
        outputs = surrogate_model(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
            labels=y
        )[1]
        if t != K - 1:
            loss_adv = F.cross_entropy(outputs, y)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        else:
            loss = loss_computer.loss(outputs, y, g, is_training, bert_pgd_loss=True)
            loss.backward()
    attack_bert_test.restore()  # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    surrogate_model.zero_grad()
    optimizer.zero_grad()

    surrogate_model.eval()

    outputs = model(x)
    loss_main = loss_computer.loss(outputs, y, g, is_training)