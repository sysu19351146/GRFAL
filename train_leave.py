def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, attack_bert=None, show_progress=True, log_every=50, scheduler=None, lamda=None, dis_cum=None,
              distance='l_inf', test_type='erm', test_eps=0.00784, test_rand=0.00196, tau=0.01):
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

    if args.model == 'bert' and not is_training and test_type == 'pgd':
        surrogate_model = model
        surrogate_model.train()
        attack_bert_test = attack_bert

    criterion_kl2 = nn.KLDivLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(size_average=False)
    loss_trade = torch.nn.CrossEntropyLoss(reduction='none')
    with torch.set_grad_enabled(is_training or (test_type == 'pgd' and args.model == 'bert')):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]

            if is_training and args.train_type == "uni":
                model.eval()
                x, delta = uni_wrm2(args, x, y, model, args.attack_iters, args.alpha_, tau, epsilon=args.epsilon,
                                    lamda=lamda, early_stop=args.early_stop)
                dis = torch.mean(mynorm(delta, order=1), dim=0)

                dis_cum.append(dis)

                model.zero_grad()
                optimizer.zero_grad()
                model.train()

            if is_training and args.train_type == "unitrades":
                model.eval()
                x_adv, delta = uni_wrm2(args, x, y, model, args.attack_iters, args.alpha_, tau, epsilon=args.epsilon,
                                        lamda=lamda, early_stop=args.early_stop)
                dis = torch.mean(mynorm(delta, order=1), dim=0)

                dis_cum.append(dis)

                model.zero_grad()
                optimizer.zero_grad()
                model.train()
                outputs = model(x)

                loss_nat = loss_trade(outputs, y)
                loss_robust = (1.0 / args.batch_size) * criterion_kl2(F.log_softmax(model(x_adv), dim=1),
                                                                      F.softmax(model(x), dim=1))

                loss_main = loss_computer.loss(outputs, y, g, is_training,
                                               trade_loss=(loss_nat, args.beta * loss_robust.sum(dim=1)))

            if is_training and args.train_type == "pgd" and args.model != 'bert':
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
                    x_v = x_v.detach() + args.epsilon / args.attack_iters * torch.sign(g_v.detach())
                    x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                    x_v = torch.clamp(x_v, args.clmin, args.clmax)
                    x_adv.data[index, :, :, :] = x_v
                # x=x_adv.detach()
                model.zero_grad()
                optimizer.zero_grad()
                model.train()
                outputs = model(x)
                loss_main = loss_computer.loss(outputs, y, g, is_training, trade_loss=(
                loss_trade(outputs, y), loss_trade(model(x_adv.detach()), y)))

            if is_training and args.train_type == "pgd" and args.model == 'bert':
                K = 3
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1]
                loss_adv = loss_computer.loss_adv(outputs, y, g, is_training, bert_pgd_loss=True)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
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
                    if t != K - 1:
                        loss_adv = loss_computer.loss_adv(outputs, y, g, is_training, bert_pgd_loss=True)
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    else:
                        loss_adv = loss_computer.loss(outputs, y, g, is_training, bert_pgd_loss=True)
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    attack_bert.backup_grad()
                attack_bert.restore()  # 恢复embedding参数
                # 梯度下降，更新参数
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()

                model.zero_grad()

            if is_training and args.train_type == "trades" and args.model == 'bert':
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
                        loss_adv = loss_computer.loss_adv(outputs, y, g, is_training, trade_loss=(
                        loss_nat, args.beta * torch.sum(loss_robust, dim=1)))
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

            if is_training and args.train_type == 'trades' and args.model != 'bert':
                model.eval()

                x_adv = x.detach() + args.random_init * torch.randn(x.shape).to(args.device).detach()
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
                            # loss_kl = loss_computer.loss_adv(output, y, g, is_training,
                            #                                trades_attack=(args.beta * torch.sum(loss_kl, dim=1)))
                        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                        x_v = x_adv[index, :, :, :]
                        g_v = grad[index, :, :, :]
                        x_o = x[index, :, :, :]
                        x_v = x_v.detach() - args.epsilon / args.attack_iters * torch.sign(g_v.detach())
                        x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                        x_v = torch.clamp(x_v, args.clmin, args.clmax)
                        x_adv.data[index, :, :, :] = x_v
                        # x_adv.grad.zero_()

                model.train()

                # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

                x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
                optimizer.zero_grad()
                outputs = model(x)
                loss_nat = loss_trade(outputs, y)
                loss_robust = (1.0 / args.batch_size) * criterion_kl2(F.log_softmax(model(x_adv), dim=1),
                                                                      F.softmax(model(x), dim=1))

                loss_main = loss_computer.loss(outputs, y, g, is_training,
                                               trade_loss=(loss_nat, args.beta * torch.sum(loss_robust, dim=1)))

            if is_training and args.train_type == 'mart':
                model.eval()
                kl = nn.KLDivLoss(reduction='none')
                x_adv = x.detach() + args.random_init * torch.randn(x.shape).to(args.device).detach()
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
                        x_v = x_adv[index, :, :, :]
                        g_v = grad[index, :, :, :]
                        x_o = x[index, :, :, :]
                        x_v = x_v.detach() + args.epsilon / args.attack_iters * torch.sign(g_v.detach())
                        x_v = torch.min(torch.max(x_v, x_o - args.epsilon), x_o + args.epsilon)
                        x_v = torch.clamp(x_v, args.clmin, args.clmax)
                        x_adv.data[index, :, :, :] = x_v
                        # x_adv.grad.zero_()

                model.train()

                # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

                x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
                optimizer.zero_grad()

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
                                               trade_loss=(loss_adv, float(args.beta) * loss_robust))

            if is_training and args.train_type == "unimart":
                model.eval()
                kl = nn.KLDivLoss(reduction='none')
                x_adv, delta = uni_wrm2(args, x, y, model, args.attack_iters, args.alpha_, tau, epsilon=args.epsilon,
                                        lamda=lamda, early_stop=args.early_stop)
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

            if not is_training and test_type == 'pgd':
                if args.model != 'bert':
                    epsilon = test_eps
                    step_size = epsilon / 10
                    random_init = test_rand
                    X_pgd = Variable(x.data, requires_grad=True)

                    random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).to(args.device)
                    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

                    for _ in range(10):
                        opt = torch.optim.SGD([X_pgd], lr=1e-3)
                        opt.zero_grad()

                        with torch.enable_grad():
                            if args.model == 'bert':
                                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                            else:
                                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                        loss.backward()
                        eta = step_size * X_pgd.grad.data.sign()
                        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                        eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
                        X_pgd = Variable(x.data + eta, requires_grad=True)
                        X_pgd = Variable(torch.clamp(X_pgd, args.clmin, args.clmax), requires_grad=True)
                    x = X_pgd
                else:

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

            if args.model == 'bert':
                if test_type != 'pgd':
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
            else:
                if (
                        args.train_type != 'trades' and args.train_type != 'unitrades' and args.train_type != 'mart' and args.train_type != 'unimart' and args.train_type != 'pgd') or not is_training:
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

            if is_training and (batch_idx + 1) % log_every == 0:
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

