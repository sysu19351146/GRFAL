import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossComputer:
    def __init__(self, criterion, is_robust, dataset, model,args,device=None,alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01,l2_norm=0,normalize_loss=False, btl=False):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.device=device
        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(self.device)
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = dataset.group_str
        self.model=model
        self.l2_norm=l2_norm
        self.args=args

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(self.device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(self.device)

        if is_robust:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(self.device)/self.n_groups
        self.adv_trade_probs=torch.ones(self.n_groups).to(self.device)/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(self.device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(self.device)
        self.adv_trade_probs_max=torch.ones(self.n_groups).to(self.device)/self.n_groups * (1+args.limit_eps)
        self.adv_trade_probs_min = torch.ones(self.n_groups).to(self.device) / self.n_groups * (1-args.limit_eps)
        self.wts=torch.exp(self.adj / torch.sqrt(self.group_counts))
        self.wts = self.wts / self.wts.sum()
        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False,trade_loss=None,bert_pgd_loss=False):
        # compute per-sample and per-group losses
        not_robust=self.args.trades_new
        if not bert_pgd_loss:
            if trade_loss!=None :
                if not_robust:
                    per_sample_losses =trade_loss[0]
                else:
                    per_sample_losses =trade_loss[0]+trade_loss[1]

            else:
                per_sample_losses = self.criterion(yhat, y)
            group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
            group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)



            # update historical losses
            self.update_exp_avg_loss(group_loss, group_count)

            # compute overall loss
            if self.is_robust and not self.btl:
                if is_training:
                    actual_loss, weights = self.compute_robust_loss_cgd(group_loss, group_count)
                else:
                    actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
                # actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
            elif self.is_robust and self.btl:
                actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
            else:
                actual_loss = per_sample_losses.mean()
                weights = None


            robust_losses=torch.tensor(0.).to(self.device)
            if trade_loss!=None and not_robust:
                robust_losses, _ = self.compute_group_avg(trade_loss[1], group_idx)

                robust_losses,_=self.compute_adversarial_robust_loss(robust_losses,group_loss)


            # update stats
            self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
            # weight_norm=torch.tensor(0.).to(self.device)
            # for w in self.model.parameters():
            #     weight_norm += w.norm().pow(2)

            #return actual_loss

            return actual_loss+robust_losses
        else:

            bert_pgd_loss=self.criterion(yhat, y)
            robust_losses, weights = self.compute_group_avg(bert_pgd_loss, group_idx)

            robust_losses, weights = self.compute_adversarial_robust_loss(robust_losses,None)

            group_loss, group_count = self.compute_group_avg(bert_pgd_loss, group_idx)
            group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)
            self.update_exp_avg_loss(robust_losses, group_count)
            self.update_stats(robust_losses, group_loss, group_acc, group_count, weights)
            return robust_losses

    def loss_adv(self, yhat, y, group_idx=None, is_training=False,trade_loss=None,bert_pgd_loss=False,trades_attack=None):
        # compute per-sample and per-group losses
        if trades_attack!=None:

            robust_losses, _ = self.compute_group_avg(trades_attack, group_idx)
            robust_losses, _ = self.compute_adversarial_robust_loss_adv(robust_losses,_)
            # update stats

            # weight_norm=torch.tensor(0.).to(self.device)
            # for w in self.model.parameters():
            #     weight_norm += w.norm().pow(2)

            # return actual_loss

            return robust_losses
        else:
            if not bert_pgd_loss:
                if trade_loss!=None :
                    per_sample_losses =trade_loss[0]

                    #per_sample_losses =trade_loss[0]+trade_loss[1]

                else:
                    per_sample_losses = self.criterion(yhat, y)
                group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
                group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

                # update historical losses
                # compute overall loss
                if self.is_robust and not self.btl:
                    actual_loss, weights = self.compute_robust_loss_adv(group_loss, group_count)
                elif self.is_robust and self.btl:
                    actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
                else:
                    actual_loss = per_sample_losses.mean()
                    weights = None


                robust_losses=torch.tensor(0.).to(self.device)
                if trade_loss!=None:
                    robust_losses, _ = self.compute_group_avg(trade_loss[1], group_idx)
                    robust_losses,_=self.compute_adversarial_robust_loss_adv(robust_losses,group_loss)
                # update stats

                # weight_norm=torch.tensor(0.).to(self.device)
                # for w in self.model.parameters():
                #     weight_norm += w.norm().pow(2)

                #return actual_loss

                return actual_loss+robust_losses
            else:

                bert_pgd_loss=self.criterion(yhat, y)
                robust_losses, weights = self.compute_group_avg(bert_pgd_loss, group_idx)

                robust_losses, weights = self.compute_adversarial_robust_loss(robust_losses,None)

                group_loss, group_count = self.compute_group_avg(bert_pgd_loss, group_idx)
                group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)
                # self.update_exp_avg_loss(robust_losses, group_count)
                # self.update_stats(robust_losses, group_loss, group_acc, group_count, weights)
                return robust_losses

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        if self.args.train_grad:
            denom = self.processed_data_counts + group_count
            denom += (denom == 0).float()
            prev_weight = self.processed_data_counts / denom
            curr_weight = group_count / denom
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            if not (self.avg_group_loss==0)[0]:
                self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data+self.step_size*1*((prev_weight*self.avg_group_loss.data + curr_weight*adjusted_loss.data-adjusted_loss.data)))
            else:
                self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            #self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            if self.args.limit_nat:
                self.adv_probs = torch.min(self.adv_probs, self.adv_trade_probs_max)
                self.adv_probs = torch.max(self.adv_probs, self.adv_trade_probs_min)
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum())
        else:
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            if self.args.limit_nat:
                self.adv_probs = torch.min(self.adv_probs, self.adv_trade_probs_max)
                self.adv_probs = torch.max(self.adv_probs, self.adv_trade_probs_min)
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
            self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_cgd(self, group_loss, group_count):

        params = []
        select = ['layer3', 'layer4', 'fc.weight', 'fc.bias']
        for name, param in self.model.named_parameters():
            for s in select:
                if (name.find(s) >= 0):
                    params.append(param)

        all_grads = [None] * self.n_groups
        for li in range(self.n_groups):
            all_grads[li] = torch.autograd.grad(group_loss[li], params, retain_graph=True)
            assert all_grads[li] is not None

        RTG = torch.zeros([self.n_groups, self.n_groups], device=self.device)
        for li in range(self.n_groups):
            for lj in range(self.n_groups):
                dp = 0
                vec1_sqnorm, vec2_sqnorm = 0, 0
                for pi in range(len(params)):
                    fvec1 = all_grads[lj][pi].detach().flatten()
                    fvec2 = all_grads[li][pi].detach().flatten()
                    dp += fvec1 @ fvec2
                    vec1_sqnorm += torch.norm(fvec1) ** 2
                    vec2_sqnorm += torch.norm(fvec2) ** 2
                RTG[li, lj] = dp / torch.clamp(torch.sqrt(vec1_sqnorm * vec2_sqnorm), min=1e-3)

        _gl = torch.sqrt(group_loss.detach().unsqueeze(-1))
        RTG = torch.mm(_gl, _gl.t()) * RTG
        _exp = self.step_size * (RTG @ self.wts)

        # to avoid overflow
        _exp -= _exp.max()
        alph = torch.exp(_exp)
        self.adv_probs *= alph.data
        self.adv_probs = self.adv_probs / self.adv_probs.sum()
        self.adv_probs = torch.clamp(self.adv_probs, min=1e-5)



        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs




    def compute_adversarial_robust_loss(self, group_loss,per_sample_losses):
        adjusted_loss = group_loss
        # if torch.all(self.adj>0):
        #     adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        # if self.normalize_loss:
        #     adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_trade_probs = self.adv_trade_probs * torch.exp(-self.step_size*adjusted_loss.data)
        if self.args.limit_adv:
            self.adv_trade_probs = torch.min(self.adv_trade_probs,self.adv_trade_probs_max)
            self.adv_trade_probs = torch.max(self.adv_trade_probs, self.adv_trade_probs_min)
        self.adv_trade_probs = self.adv_trade_probs/(self.adv_trade_probs.sum())


        robust_loss = group_loss @ (self.adv_trade_probs*4)
        return robust_loss, self.adv_trade_probs

    def compute_robust_loss_adv(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # denom = self.processed_data_counts + group_count
        # denom += (denom == 0).float()
        # prev_weight = self.processed_data_counts / denom
        # curr_weight = group_count / denom
        # #adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        # # if not (self.avg_group_loss==0)[0]:
        # #     adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data+self.step_size*1*(1-(prev_weight*self.avg_group_loss.data + curr_weight*adjusted_loss.data)/self.avg_group_loss.data))
        # # else:
        # #     adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        # adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        # # adv_probs = torch.min(adv_probs, self.adv_trade_probs_max)
        # # adv_probs = torch.max(adv_probs, self.adv_trade_probs_min)
        # #adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        # adv_probs = adv_probs/(adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_adversarial_robust_loss_adv(self, group_loss,per_sample_losses):
        #adjusted_loss = group_loss
        # if torch.all(self.adj>0):
        #     adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        # if self.normalize_loss:
        #     adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # adv_trade_probs = self.adv_trade_probs * torch.exp(-self.step_size*adjusted_loss.data)
        # adv_trade_probs = torch.min(self.adv_trade_probs,self.adv_trade_probs_max)
        # adv_trade_probs = torch.max(self.adv_trade_probs, self.adv_trade_probs_min)
        # adv_trade_probs = self.adv_trade_probs/(self.adv_trade_probs.sum())


        robust_loss = group_loss @ (self.adv_trade_probs*4)
        return robust_loss, self.adv_trade_probs



    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
        weights = mask.float() * sorted_frac /self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.
        #self.adv_probs = torch.ones(self.n_groups).to(self.device) / self.n_groups

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        #logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'trades prob = {self.adv_trade_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()

    def turn_robust(self):
        self.is_robust=False


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = 0.01
    if epoch >= 100:
        lr = lr * 0.001
    elif epoch >= 90:
        lr = lr * 0.01
    elif epoch >= 75:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr