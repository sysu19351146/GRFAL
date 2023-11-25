import copy
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule
from WRM import *
from models import *




def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, attack_bert=None,show_progress=True, log_every=50, scheduler=None,lamda=None,dis_cum=None,distance='l_inf',test_type='erm',test_eps=0.01,test_rand=0.01,tau=0.01):
    """
    scheduler is only used inside this function if model is bert.
    """



    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    if args.model=='bert' and not is_training and test_type=='pgd':
        surrogate_model=model
        surrogate_model.train()
        attack_bert_test=attack_bert


    criterion_kl2 = nn.KLDivLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss_trade=torch.nn.CrossEntropyLoss(reduction='none')
    with torch.set_grad_enabled(is_training or (test_type=='pgd' and args.model=='bert')):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]


            if is_training and args.train_type=="uni":
                model.eval()
                x, delta = uni_wrm2(args,x, y, model, args.attack_iters,args.alpha_,tau,epsilon=args.epsilon, lamda=lamda,early_stop=args.early_stop)
                dis = torch.mean(mynorm(delta, order=1), dim=0)

                dis_cum.append(dis)

                model.zero_grad()
                optimizer.zero_grad()
                model.train()

            if is_training and args.train_type=="unitrades":
                model.eval()
                x_adv, delta = uni_wrm2(args,x, y, model, args.attack_iters,args.alpha_,tau,epsilon=args.epsilon, lamda=lamda,early_stop=args.early_stop)
                dis = torch.mean(mynorm(delta, order=1), dim=0)

                dis_cum.append(dis)

                model.zero_grad()
                optimizer.zero_grad()
                model.train()
                outputs = model(x)

                loss_nat =loss_trade(outputs, y)
                loss_robust = (1.0 / args.batch_size) * criterion_kl2(F.log_softmax(model(x_adv), dim=1),
                                                                     F.softmax(model(x), dim=1))

                loss_main = loss_computer.loss(outputs, y, g, is_training,trade_loss=(loss_nat , args.beta * loss_robust.sum(dim=1)))
            
            if is_training and args.train_type=="pgd" and args.model!= 'bert':
                model.eval()
                x_adv = x.detach() + args.random_init * torch.randn(x.shape).to(args.device).detach()
                for _ in range(args.attack_iters):
                    x_adv.requires_grad_()
                    output = model(x_adv)
                    if args.early_stop:
                        index = torch.where(output.max(1)[1] == y)[0]
                    else:
                        index = slice(None, None, None)
                    with torch.enable_grad():
                        loss_ce = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                    x_v = x_adv[index, :, :, :]
                    g_v = grad[index, :, :, :]
                    x_o = x[index, :, :, :]
                    x_v = x_v.detach() + args.epsilon/args.attack_iters * torch.sign(g_v.detach())
                    x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                    x_v = torch.clamp(x_v, args.clmin, args.clmax)
                    x_adv.data[index, :, :, :] = x_v
                x=x_adv.detach()
                model.zero_grad()
                optimizer.zero_grad()
                model.train()
                # outputs = model(x)
                # loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training and args.train_type=="pgd" and args.model== 'bert' :
                K=3
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1]
                loss=F.cross_entropy(outputs,y)
                loss.backward()  # 反向传播，得到正常的grad
                attack_bert.backup_grad()
                # 对抗训练
                for t in range(K):
                    attack_bert.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.processor
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        attack_bert.restore_grad()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=y
                    )[1]
                    loss_adv = loss_computer.loss(outputs, y, g, is_training,bert_pgd_loss=True)
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                attack_bert.restore()  # 恢复embedding参数
                # 梯度下降，更新参数
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()

                model.zero_grad()

            if is_training and args.train_type=='trades':
                model.eval()

                x_adv = x.detach() +  args.random_init * torch.randn(x.shape).to(args.device).detach()
                if distance == 'l_inf':
                    for _ in range(args.attack_iters):
                        output = model(x_adv)
                        x_adv.requires_grad_()
                        if args.early_stop:
                            index = torch.where(output.max(1)[1] == y)[0]
                        else:
                            index = slice(None, None, None)




                        with torch.enable_grad():
                            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                   F.softmax(model(x), dim=1))
                        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                        x_v=x_adv[index, :, :, :]
                        g_v=grad[index, :, :, :]
                        x_o=x[index, :, :, :]
                        x_v = x_v.detach() - args.epsilon/args.attack_iters * torch.sign(g_v.detach())
                        x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                        x_v = torch.clamp(x_v, args.clmin, args.clmax)
                        x_adv.data[index, :, :, :]=x_v
                        #x_adv.grad.zero_()

                model.train()

                # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

                x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
                optimizer.zero_grad()
                outputs = model(x)
                loss_nat = loss_trade(outputs, y)
                loss_robust = (1.0 / args.batch_size) * criterion_kl2(F.log_softmax(model(x_adv), dim=1),
                                                                      F.softmax(model(x), dim=1))

                loss_main = loss_computer.loss(outputs, y, g, is_training,
                                               trade_loss=(loss_nat , args.beta * loss_robust.sum(dim=1)))


            if is_training and args.train_type=='mart':
                model.eval()
                kl = nn.KLDivLoss(reduction='none')
                x_adv = x.detach() +  args.random_init * torch.randn(x.shape).to(args.device).detach()
                if distance == 'l_inf':
                    for _ in range(args.attack_iters):
                        output = model(x_adv)
                        x_adv.requires_grad_()
                        if args.early_stop:
                            index = torch.where(output.max(1)[1] == y)[0]
                        else:
                            index = slice(None, None, None)




                        with torch.enable_grad():
                            loss_ce = F.cross_entropy(model(x_adv), y)
                        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                        x_v=x_adv[index, :, :, :]
                        g_v=grad[index, :, :, :]
                        x_o=x[index, :, :, :]
                        x_v = x_v.detach() + args.epsilon/args.attack_iters * torch.sign(g_v.detach())
                        x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                        x_v = torch.clamp(x_v, args.clmin, args.clmax)
                        x_adv.data[index, :, :, :]=x_v
                        #x_adv.grad.zero_()

                model.train()

                # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

                x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
                optimizer.zero_grad()


                logits = model(x)

                logits_adv = model(x_adv)

                adv_probs = F.softmax(logits_adv, dim=1)

                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

                loss_adv = F.cross_entropy(logits_adv, y,reduce=False) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y,reduce=False)

                nat_probs = F.softmax(logits, dim=1)

                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

                loss_robust = (1.0 / args.batch_size) * (
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                #loss = loss_adv + float(args.beta) * loss_robust


                loss_main = loss_computer.loss(logits_adv, y, g, is_training,
                                               trade_loss=(loss_adv , float(args.beta) * loss_robust))

            if is_training and args.train_type=="unimart":
                model.eval()
                kl = nn.KLDivLoss(reduction='none')
                x_adv, delta = uni_wrm2(args,x, y, model, args.attack_iters,args.alpha_,tau,epsilon=args.epsilon, lamda=lamda,early_stop=args.early_stop)
                dis = torch.mean(mynorm(delta, order=1), dim=0)

                dis_cum.append(dis)

                model.zero_grad()
                optimizer.zero_grad()
                model.train()


                logits = model(x)

                logits_adv = model(x_adv)

                adv_probs = F.softmax(logits_adv, dim=1)

                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

                loss_adv = F.cross_entropy(logits_adv, y, reduce=False) + F.nll_loss(
                    torch.log(1.0001 - adv_probs + 1e-12), new_y, reduce=False)

                nat_probs = F.softmax(logits, dim=1)

                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

                loss_robust = (1.0 / args.batch_size) * (
                        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                # loss = loss_adv + float(args.beta) * loss_robust

                loss_main = loss_computer.loss(logits_adv, y, g, is_training,
                                               trade_loss=loss_adv + float(args.beta) * loss_robust)


            if not is_training and test_type=='pgd':
                if args.model !='bert':
                    epsilon=test_eps
                    step_size=epsilon/10
                    random_init=test_rand
                    X_pgd = Variable(x.data, requires_grad=True)

                    random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).to(args.device)
                    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

                    for _ in range(10):
                        opt = torch.optim.SGD([X_pgd], lr=1e-3)
                        opt.zero_grad()

                        with torch.enable_grad():
                            if args.model=='bert':
                                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                            else:
                                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                        loss.backward()
                        eta = step_size * X_pgd.grad.data.sign()
                        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                        eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
                        X_pgd = Variable(x.data + eta, requires_grad=True)
                        X_pgd = Variable(torch.clamp(X_pgd, args.clmin, args.clmax), requires_grad=True)
                    x=X_pgd
                else:

                    surrogate_model.train()
                    surrogate_model.zero_grad()

                    K = 3
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = surrogate_model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=y
                    )[1]


                    loss= F.cross_entropy(outputs,y)

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
                        if t!=100:
                            loss_adv = F.cross_entropy(outputs, y)
                            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        else:
                            loss=loss_computer.loss(outputs, y, g, is_training)
                    attack_bert_test.restore()  # 恢复embedding参数
                    # 梯度下降，更新参数
                    optimizer.step()
                    surrogate_model.zero_grad()
                    optimizer.zero_grad()

                    surrogate_model.eval()



            if args.model == 'bert'  :
                if test_type!='pgd':
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids,
                        labels=y
                    )[1] # [1] returns logits
                    loss_main = loss_computer.loss(outputs, y, g, is_training)
            else:
                if (args.train_type !='trades' and args.train_type !='unitrades' and args.train_type !='mart' and args.train_type !='unimart' ) or not is_training:
                    outputs = model(x)
                    loss_main = loss_computer.loss(outputs, y, g, is_training)







            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()



        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.to(args.device)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        model=model,
        device=args.device,
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        l2_norm=args.l2_norm)

    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
        attack_bert=Bert_attack(model)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        attack_bert = None

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    best_test_acc = 0
    test_type='erm'

    if not args.only_test:
        if args.train_type=='uni'  or args.train_type=='unitrades' or args.train_type=='unimart':
            lamda = torch.tensor(args.lamda, requires_grad=False).to(args.device)
            dis_cum = []
            tau = args.tau
            lr_tau = args.lr_tau

        else:
            lamda=None
            dis_cum=None
            tau=0



        for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
            logger.write('\nEpoch [%d]:\n' % epoch)
            logger.write(f'Training:\n')
            if epoch==5 and args.is_combine:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.fc.parameters()),
                    lr=optimizer.param_groups[0]['lr'],
                    momentum=optimizer.param_groups[0]['momentum'],
                    weight_decay=optimizer.param_groups[0]['weight_decay'])
                train_loss_computer.turn_robust()
                test_type='pgd'
                args.train_type='pgd'

            run_epoch(
                epoch, model, optimizer,
                dataset['train_loader'],
                train_loss_computer,
                logger, train_csv_logger, args,
                attack_bert=attack_bert,
                is_training=True,
                show_progress=True,
                log_every=args.log_every,
                scheduler=scheduler,
                lamda=lamda,
                dis_cum=dis_cum,
                tau=tau)



            if args.train_type=='uni' or args.train_type=='unitrades'or args.train_type=='unimart' :
                tau = tau + lr_tau * tau
                # if epoch % 3 == 0:
                #     lamda = lamda + 0.02 * (args.epsilon - sum(dis_cum) / len(dis_cum))
                #     dis_cum = []

            logger.write(f'\nValidation:\n')
            val_loss_computer = LossComputer(
                criterion,
                device=args.device,
                is_robust=args.robust,
                dataset=dataset['val_data'],
                model=model,
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['val_loader'],
                val_loss_computer,
                logger, val_csv_logger, args,
                is_training=False,
                test_type=test_type)

            # Test set; don't print to avoid peeking
            if dataset['test_data'] is not None:
                test_loss_computer = LossComputer(
                    criterion,
                    device=args.device,
                    is_robust=args.robust,
                    dataset=dataset['test_data'],
                    model=model,
                    step_size=args.robust_step_size,
                    alpha=args.alpha)
                run_epoch(
                    epoch, model, optimizer,
                    dataset['test_loader'],
                    test_loss_computer,
                    logger, test_csv_logger, args,
                    is_training=False,
                    attack_bert=attack_bert,
                    test_type=args.test_type

                    )

            # Inspect learning rates
            if (epoch+1) % 1 == 0:
                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']
                    logger.write('Current lr: %f\n' % curr_lr)

            if args.scheduler and args.model != 'bert':
                if args.robust:
                    val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
                else:
                    val_loss = val_loss_computer.avg_actual_loss
                scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

            if epoch % args.save_step == 0 or epoch<=1:
                torch.save(model, os.path.join(args.log_dir, args.name_index+'%d_model.pth' % epoch))

            if args.save_last:
                torch.save(model, os.path.join(args.log_dir, args.name_index+'last_model.pth'))

            if args.save_best:
                if args.robust or args.reweight_groups:
                    curr_val_acc = min(val_loss_computer.avg_group_acc)
                else:
                    curr_val_acc = val_loss_computer.avg_acc
                logger.write(f'Current validation accuracy: {curr_val_acc}\n')
                if curr_val_acc > best_val_acc:
                    best_val_acc = curr_val_acc
                    torch.save(model, os.path.join(args.log_dir, args.name_index+'best_model.pth'))
                    logger.write(f'Best model saved at epoch {epoch}\n')
                if args.robust or args.reweight_groups:
                    curr_test_acc = min(test_loss_computer.avg_group_acc)
                else:
                    curr_test_acc = test_loss_computer.avg_acc
                logger.write(f'Current validation accuracy: {curr_test_acc}\n')
                if curr_test_acc > best_test_acc:
                    best_test_acc = curr_test_acc
                    torch.save(model, os.path.join(args.log_dir, args.name_index+'best_test_model.pth'))
                    logger.write(f'Best test model saved at epoch {epoch}\n')

            if args.automatic_adjustment:
                gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
                adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
                train_loss_computer.adj = adjustments
                logger.write('Adjustments updated\n')
                for group_idx in range(train_loss_computer.n_groups):
                    logger.write(
                        f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                        f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
            logger.write('\n')
    if args.only_test:
        for i in range(10):
            if dataset['test_data'] is not None:

                test_loss_computer = LossComputer(
                    criterion,
                    is_robust=args.robust,
                    dataset=dataset['test_data'],
                    model=model,
                    device=args.device,
                    step_size=args.robust_step_size,
                    alpha=args.alpha
                )
                if i==0:
                    run_epoch(
                        0, model, optimizer,
                        dataset['test_loader'],
                        test_loss_computer,
                        logger, test_csv_logger, args,
                        is_training=False,
                        test_type='erm',
                        test_eps=i * 0.01,
                        test_rand=i * 0.003)
                else:
                    run_epoch(
                        0, model, optimizer,
                        dataset['test_loader'],
                        test_loss_computer,
                        logger, test_csv_logger, args,
                        is_training=False,
                        test_type='pgd',
                        test_eps=i*0.01,
                        test_rand=i*0.003)
