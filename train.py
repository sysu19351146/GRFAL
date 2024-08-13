import copy
import os
import types
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer,adjust_learning_rate

from pytorch_transformers import AdamW, WarmupLinearSchedule
from WRM import *
from models import *
from method import *
import torchvision


def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, attack_bert=None,show_progress=True, log_every=50, scheduler=None,lamda=None,dis_cum=None,distance='l_inf',test_type='erm',test_eps=0.00196,test_rand=0.0001,tau=0.01):
    """
    scheduler is only used inside this function if model is bert.
    """

    no_log=['WAT','BAT','CFA','DAFA','FAT']

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

    with torch.set_grad_enabled(is_training or (test_type=='pgd' and args.model=='bert')):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            batch_id=batch[3]

            
            if is_training and args.train_type=="pgd" and args.model!= 'bert':
                pgd_train(model,x,y,g,is_training,loss_computer,args,optimizer)

            if is_training and args.train_type=="pgd" and args.model== 'bert' :
                pgd_bert_train(model,x,y,g,is_training,loss_computer,args,optimizer,scheduler,attack_bert)

            if is_training and args.train_type=="trades" and args.model== 'bert' :
                trades_bert_train(model,x,y,g,is_training,loss_computer,args,optimizer,scheduler,attack_bert)


            if is_training and args.train_type=='trades' and args.model!= 'bert' :
                trades_train(model,x,y,g,is_training,loss_computer,args,optimizer)



            if not is_training and test_type=='pgd':
                if args.model !='bert':
                    pgd_test(model,x,y,g,is_training,loss_computer,args,test_eps,test_rand)
                else:
                    pgd_bert_test(model,surrogate_model,x,y,g,is_training,loss_computer,optimizer,attack_bert_test,test_eps,test_rand)

            if not is_training and test_type=='AA':
                if args.model !='bert':
                    AA_test(model,x,y,g,is_training,loss_computer,args,test_eps,test_rand)


            if is_training and args.train_type=='BAT':
                BAT(model,x,y,g,is_training,loss_computer,args,optimizer)

            if is_training and args.train_type=='WAT':
                WAT(model,x,y,g,is_training,loss_computer,args,optimizer)

            if is_training and args.train_type=='CFA':
                loss, output = CFA(model, x, y,g, args.CFA_log.class_eps,args.CFA_log.class_beta,args.CFA_log.alpha,args.attack_iters )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                args.CFA_log.update_robust(output, y,g)

                clean_output = model(x).detach()
                args.CFA_log.update_clean(clean_output, y,g)

            if is_training and args.train_type=='DAFA':
                DAFA(model=model, x_natural=x, y=y, g=y,optimizer=optimizer, args=args,
                                              class_weights=args.class_weights, batch_indices=batch_id,
                                              memory_dict=args.memory_dict)

            if is_training and args.train_type=='FAT':
                FAT(model,x,y,g,args,optimizer)


            if not is_training and test_type=='erm':
                outputs = model(x)
                loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training and args.train_type=='erm':
                outputs = model(x)
                loss_main = loss_computer.loss(outputs, y, g, is_training)
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0 and args.train_type not in no_log:
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
        args=args,
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
        attack_bert=Bert_attack(model,args.device)
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
    SEAT_init=False
    test_type='pgd'
    FAWA_model=torchvision.models.resnet50(pretrained=True)
    d = FAWA_model.fc.in_features
    FAWA_model.fc = nn.Linear(d, dataset['train_data'].n_classes)
    FAWA_model=FAWA_model.to(args.device)
    if args.train_type=='erm':
        test_type = 'erm'

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

        if args.train_type=='WAT':
            args.nat_class_weights = torch.ones(args.class_num + 1).to(args.device)
            args.bndy_class_weights = torch.ones(args.class_num + 1).to(args.device)
            wat_valid_nat_cost = torch.zeros(args.n_epochs, args.class_num + 1).to(args.device)
            wat_valid_bndy_cost = torch.zeros(args.n_epochs, args.class_num + 1).to(args.device)
            wat_valid_cost = torch.zeros(args.n_epochs, args.class_num + 1).to(args.device)

        if args.train_type=='CFA':
            args.CFA_log= CW_log(args.device,args.class_num)
            args.CFA_log.eps = args.epsilon
            args.CFA_log.alpha = args.epsilon/4  
            args.CFA_log.beta = args.beta
            args.CFA_log.class_eps = torch.ones(args.class_num).to(args.device) * args.CFA_log.eps
            args.CFA_log.class_beta = torch.ones(args.class_num).to(args.device) * (args.CFA_log.beta / (1 +args.CFA_log.beta))

        if args.train_type=='DAFA':
            args.class_weights = torch.ones(args.n_classes).to(args.device)

        if args.train_type=='FAT':
            rate = 0.02
            delta0 = 0.1 * torch.ones(args.class_num)
            delta1 = 0.15 * torch.ones(args.class_num)
            lmbda = torch.zeros(args.class_num * 3)
            args.FAT_args=FAT_args()



        for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
            epoch_init_time = time.time()
            logger.write('\nEpoch [%d]:\n' % epoch)
            logger.write(f"Time:{epoch_init_time:.4f} |\n")
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

            if args.dataset=='Cifar':
                adjust_learning_rate(optimizer, epoch)

            if args.train_type == 'DAFA' and epoch == 0:
                args.memory_dict = {'probs': [],
                               'labels': []}
            else:
                args.memory_dict = None

            if args.train_type =='FAT':
                class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
                    test_valid_FAT(model, dataset['val_loader'], args)
                gamma0 = class_clean_error - total_clean_error - delta0
                gamma1 = class_bndy_error - total_bndy_error - delta1

                if rate % 30 == 0:
                    rate = rate / 2

                ## constraints coefficients
                lmbda0 = lmbda[0:args.class_num] + rate / 5 * torch.clamp(gamma0, min=0)
                lmbda0 = torch.clamp(lmbda0, min=0)
                lmbda1 = lmbda[args.class_num:args.class_num*2] + rate / 5 * torch.clamp(gamma1, min=0)
                lmbda1 = torch.clamp(lmbda1, min=0)
                lmbda2 = lmbda[args.class_num*2:args.class_num*3] + 2 * rate * (gamma1)
                lmbda2 = torch.clamp(lmbda2, min=-0.2, max=0.4)
                lmbda = torch.cat([lmbda0, lmbda1, lmbda2])

                args.FAT_args.diff0, args.FAT_args.diff1, args.FAT_args.diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2,args.class_num)


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

            logger.write(f"Time:{(time.time() - epoch_init_time):.4f} |\n")

            if args.train_type=='uni' or args.train_type=='unitrades'or args.train_type=='unimart' :
                tau = tau + lr_tau * tau

            if args.train_type=='WAT':

                wat_valid_nat_cost, wat_valid_bndy_cost, val_rob_acc, val_rob_class_wise_acc = validate(model,
                                                                                                        dataset['val_loader'],
                                                                                                        wat_valid_nat_cost,
                                                                                                        wat_valid_bndy_cost,
                                                                                                        args,
                                                                                                        epoch,
                                                                                                        device=args.device)
                for i in range(args.class_num + 1):
                    wat_valid_cost[epoch, i] = wat_valid_nat_cost[epoch, i] + args.beta * wat_valid_bndy_cost[
                        epoch, i]
                    class_factor = (torch.sum(wat_valid_cost, dim=0) * args.eta).exp()
                    args.nat_class_weights = args.class_num * class_factor / class_factor.sum()
                    args.bndy_class_weights = args.class_num * class_factor / class_factor.sum()
            elif args.train_type=='CFA':
                train_result=args.CFA_log.result()

                if epoch >= 5:
                    train_robust = train_result[3].to(args.device)
                    args.CFA_log.class_eps = (torch.ones(args.class_num).to(args.device) * 0.5 + train_robust) * args.CFA_log.eps
                else:
                    args.CFA_log.class_eps = torch.ones(args.class_num).to(args.device) * args.CFA_log.eps

                if  epoch >= 5:
                    for i in range(args.class_num):

                        args.CFA_log.class_beta[i] = (0.5 + train_robust[i]) * args.CFA_log.beta / (1 + (0.5 + train_robust[i]) * args.CFA_log.beta)
                else:
                    args.CFA_log.class_beta = torch.ones(args.class_num).to(args.device) * (args.CFA_log.beta / (1 + args.CFA_log.beta))

            elif args.train_type == 'DAFA' and epoch == 0:
                args.class_weights = calculate_class_weights(args.memory_dict, 1.5)
            # elif args.train_type=='FAT':
            #     Hamiltonian_func = Hamiltonian(h_net.layer_one, 5e-4)
            #     layer_one_optimizer = torch.optim.SGD(h_net.layer_one.parameters(), lr=lr_scheduler.get_lr()[0], momentum=0.9,
            #                                     weight_decay=5e-4)
            #     layer_one_optimizer_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(layer_one_optimizer,
            #                                                                       milestones=[80, 100, 120], gamma=0.2)
            #     LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, layer_one_optimizer,
            #                                                   args.inner_iters, sigma=2 / 255, eps=8 / 255)
            logger.write(f"Time:{(time.time() - epoch_init_time):.4f} |\n")
            logger.write(f'\nValidation_adv:\n')
            val_loss_computer = LossComputer(
                criterion,
                args=args,
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

            logger.write(f'\nValidation_nat:\n')
            val_loss_computer = LossComputer(
                criterion,
                args=args,
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
                test_type='erm')

            logger.write(f"Time:{(time.time() - epoch_init_time):.4f} |\n")

            if args.train_type=='CFA':
                R_min = min(val_loss_computer.avg_group_acc)
                if R_min >= 0.25:
                    if not SEAT_init:
                        SEAT_init = True
                        weight_average(FAWA_model, model,0.88, True)
                    else:
                        weight_average(FAWA_model, model, 0.88, False)
                else:
                    weight_average(FAWA_model, model, 1., False)


            # Test set; don't print to avoid peeking
            if dataset['test_data'] is not None:
                if args.train_type!='CFA':
                    test_loss_computer = LossComputer(
                        criterion,
                        args=args,
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
                        test_type='erm'
                    )
                    if args.test_type!='erm':
                        test_loss_computer = LossComputer(
                            criterion,
                            args=args,
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
                if args.train_type=='CFA':
                    test_loss_computer = LossComputer(
                        criterion,
                        args=args,
                        device=args.device,
                        is_robust=args.robust,
                        dataset=dataset['test_data'],
                        model=FAWA_model,
                        step_size=args.robust_step_size,
                        alpha=args.alpha)
                    run_epoch(
                        epoch, FAWA_model, optimizer,
                        dataset['test_loader'],
                        test_loss_computer,
                        logger, test_csv_logger, args,
                        is_training=False,
                        attack_bert=attack_bert,
                        test_type='erm'
                    )
                    test_loss_computer = LossComputer(
                        criterion,
                        args=args,
                        device=args.device,
                        is_robust=args.robust,
                        dataset=dataset['test_data'],
                        model=FAWA_model,
                        step_size=args.robust_step_size,
                        alpha=args.alpha)
                    run_epoch(
                        epoch, FAWA_model, optimizer,
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
                    curr_val_acc = min(val_loss_computer.avg_group_acc)
                logger.write(f'Current validation accuracy: {curr_val_acc}\n')
                if curr_val_acc > best_val_acc:
                    best_val_acc = curr_val_acc
                    torch.save(model, os.path.join(args.log_dir, args.name_index+'best_model.pth'))
                    logger.write(f'Best model saved at epoch {epoch}\n')
                if args.robust or args.reweight_groups:
                    curr_test_acc = min(test_loss_computer.avg_group_acc)
                else:
                    curr_test_acc = min(test_loss_computer.avg_group_acc)
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
        # for i in range(10):
        #     if dataset['test_data'] is not None:
        #
        #         test_loss_computer = LossComputer(
        #             criterion,
        #             args=args,
        #             device=args.device,
        #             is_robust=args.robust,
        #             dataset=dataset['test_data'],
        #             model=model,
        #             step_size=args.robust_step_size,
        #             alpha=args.alpha)
        #
        #         if i==0:
        #             run_epoch(
        #                 0, model, optimizer,
        #                 dataset['test_loader'],
        #                 test_loss_computer,
        #                 logger, test_csv_logger, args,
        #                 is_training=False,
        #                 attack_bert=None,
        #                 test_type='erm'
        #             )
        #         else:
        #             run_epoch(
        #                 0, model, optimizer,
        #                 dataset['test_loader'],
        #                 test_loss_computer,
        #                 logger, test_csv_logger, args,
        #                 is_training=False,
        #                 test_type='pgd',
        #                 attack_bert=None,
        #                 test_eps=i*0.00196,
        #                 )

        test_loss_computer = LossComputer(
            criterion,
            args=args,
            device=args.device,
            is_robust=args.robust,
            dataset=dataset['test_data'],
            model=model,
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            0, model, optimizer,
            dataset['test_loader'],
            test_loss_computer,
            logger, test_csv_logger, args,
            is_training=False,
            attack_bert=None,
            test_type=args.test_type
        )

