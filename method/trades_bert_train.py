import torch
import torch.nn as nn
criterion_kl2 = nn.KLDivLoss(reduction='none')
criterion_kl = nn.KLDivLoss(size_average=False)
loss_trade = nn.CrossEntropyLoss(reduction='none')


def trades_bert_train(model,x,y,g,is_training,loss_computer,args,optimizer,scheduler,attack_bert):
    K = 3
    input_ids = x[:, :, 0]
    input_masks = x[:, :, 1]
    segment_ids = x[:, :, 2]
    # outputs = model(
    #     input_ids=input_ids,
    #     attention_mask=input_masks,
    #     token_type_ids=segment_ids,
    #     labels=y
    # )[1]
    #
    # loss_adv = loss_computer.loss_adv(outputs, y, g, is_training, bert_pgd_loss=True)
    # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    # attack_bert.backup_grad()
    attack_bert.save_origin()
    attack_bert.random()

    # 对抗训练
    for t in range(K):
        # 在embedding上添加对抗扰动, first attack时备份param.processor
        if t != K - 1:
            model.zero_grad()
        else:
            attack_bert.restore_grad()
        attack_bert.save_attack()
        attack_bert.restore_origin()
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
            labels=y
        )[1]
        attack_bert.restore_attack()
        outputs_adv = model(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids,
            labels=y
        )[1]
        if t != K - 1:
            loss_nat = loss_trade(outputs, y)
            loss_robust = criterion_kl2(F.log_softmax(outputs_adv, dim=1),
                                        F.softmax(outputs, dim=1))
            loss_adv = loss_computer.loss_adv(outputs, y, g, is_training,
                                              trade_loss=(loss_nat, args.beta * torch.sum(loss_robust, dim=1)))
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        else:
            loss_nat = loss_trade(outputs, y)
            loss_robust = criterion_kl2(F.log_softmax(outputs_adv, dim=1),
                                        F.softmax(outputs, dim=1))
            loss_adv = loss_computer.loss(outputs, y, g, is_training,
                                          trade_loss=(loss_nat, args.beta * torch.sum(loss_robust, dim=1)))
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        attack_bert.backup_grad()
        if t != K - 1:
            attack_bert.attack()

    attack_bert.restore()  # 恢复embedding参数
    # 梯度下降，更新参数
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scheduler.step()
    optimizer.step()

    model.zero_grad()

    input_ids = x[:, :, 0]
    input_masks = x[:, :, 1]
    segment_ids = x[:, :, 2]
    outputs = model(
        input_ids=input_ids,
        attention_mask=input_masks,
        token_type_ids=segment_ids,
        labels=y
    )[1]  # [1] returns logits
    loss_main = loss_computer.loss(outputs, y, g, is_training)

    loss_main.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    scheduler.step()
    optimizer.step()
    model.zero_grad()

