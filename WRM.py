# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:14:57 2022

@author: 53412
"""
#from model import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# def define_loss():
#     Loss=torch.nn.CrossEntropyLoss()
#     return Loss


# def define_optimizer():
#     learnig_rate=1e-4
#     optimizer=torch.optim.Adam(net.parameters(),lr=learnig_rate)
#     return optimizer



class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return torch.mean(x-y)

my_loss=My_loss()
loss_l2 = nn.MSELoss(reduction='none')

def prox_z_maxizer(data0,label,step_t,net,max_step,Loss,opt,alpha,is_vector):
    opt.zero_grad()
    if is_vector:
        iter_data=data0.clone().detach().requires_grad_(True)
        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output.unsqueeze(0),label.unsqueeze(0))
            loss.backward()
            iter_data_1=iter_data+iter_data.grad*step_t
            opt.zero_grad()
            
            
            iter_data_1.view(1,-1)
            sort_data,_=torch.sort(torch.abs(iter_data_1-data0),descending=True)
            length=sort_data.shape[0]
            max_j=0
            for j in range(length):
                if torch.sum(sort_data[:j])-(1/(alpha*step_t)+(j+1-1))*sort_data[j]<0:
                    max_j=j
                else:
                    break
            belta=(1/(alpha*step_t*(max_j+1)))*torch.sum(sort_data[:max_j+1])
                
            iter_data_2=iter_data_1 - F.relu(torch.abs(iter_data_1-data0)-belta)*torch.sign(iter_data_1-data0)
            iter_data=iter_data_2.clone().detach().requires_grad_(True)
    else:
        iter_data=data0.clone().detach().unsqueeze(0).requires_grad_(True)
        base_data=data0.clone().detach().view(1,-1)
        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label.unsqueeze(0))
            loss.backward()
            iter_data_1=iter_data+iter_data.grad/torch.norm(iter_data.grad)*step_t
            # iter_data_1=iter_data+iter_data.grad*step_t
            opt.zero_grad()
            
            
            iter_data_1=iter_data_1.view(1,-1)
            sort_data,_=torch.sort(torch.abs(iter_data_1-base_data),descending=True)
            sort_data=torch.flatten(sort_data,0)
            length=sort_data.shape[0]
            max_j=0
            for j in range(length):
                if torch.sum(sort_data[:j])-(1/(alpha*step_t)+(j+1-1))*sort_data[j]<0:
                    max_j=j
                else:
                    break
            belta=(1/(alpha*step_t*(max_j+1)))*torch.sum(sort_data[:max_j+1])
            #belta=torch.max(belta,torch.tensor(1))
            #iter_data_2=iter_data_1 - torch.max(F.relu(torch.abs(iter_data_1-base_data)-belta),torch.tensor(0.1))*torch.sign(iter_data_1-base_data)
            iter_data_2=iter_data_1 - F.relu(torch.abs(iter_data_1-base_data)-belta)*torch.sign(iter_data_1-base_data)
            iter_data_2=iter_data_2.view(1,1,iter_data.shape[2],-1)
            iter_data=iter_data_2.clone().detach().requires_grad_(True)
        net.zero_grad()
        
    return iter_data

def prox_z_maxizer_inf2(data0,label,step_t,net,max_step,Loss,opt,alpha,is_vector):
    opt.zero_grad()
    if is_vector:
        iter_data=data0.clone().detach().requires_grad_(True)
        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()
            gap=torch.abs(iter_data-data0)
            s_,_=torch.max(gap,1)
            s_=s_.unsqueeze(1).repeat(1,gap.shape[1])
            iter_data_1=iter_data+iter_data.grad*step_t-step_t*s_*(gap==s_)*alpha
            opt.zero_grad()
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
            
            # iter_data_1.view(1,-1)
            # sort_data,_=torch.sort(torch.abs(iter_data_1-data0),descending=True)
            # length=sort_data.shape[0]
            # max_j=0
            # for j in range(length):
            #     if torch.sum(sort_data[:j])-(1/(alpha*step_t)+(j+1-1))*sort_data[j]<0:
            #         max_j=j
            #     else:
            #         break
            # belta=(1/(alpha*step_t*(max_j+1)))*torch.sum(sort_data[:max_j+1])
                
            # iter_data_2=iter_data_1 - F.relu(torch.abs(iter_data_1-data0)-belta)*torch.sign(iter_data_1-data0)
            # iter_data=iter_data_2.clone().detach().requires_grad_(True)
    else:
        iter_data=data0.clone().detach().requires_grad_(True)

        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()
            #iter_data_1=iter_data+iter_data.grad/torch.norm(iter_data.grad)*step_t
            gap=torch.abs(iter_data_1-data0)
            s_,_=torch.max(gap.view(gap.shape[0],-1),1)
            s_=s_.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,gap.shape[2],gap.shape[3])
            iter_data_1=torch.clamp(iter_data+iter_data.grad*step_t-step_t*s_*(gap==s_)*alpha, min=0, max=1)
            opt.zero_grad()
            
            
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
        net.zero_grad()
        
    return iter_data


def prox_z_maxizer_inf3(data0,label,step_t,net,max_step,Loss,opt,alpha,is_vector):
    opt.zero_grad()
    if is_vector:
        iter_data=data0.clone().detach().requires_grad_(True)
        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()
            gap=torch.abs(iter_data-data0)
            s_,_=torch.max(gap,1)
            s_=s_.unsqueeze(1).repeat(1,gap.shape[1])
            iter_data_1=iter_data+iter_data.grad*step_t-step_t*s_*(gap==s_)*alpha
            opt.zero_grad()
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
            
            # iter_data_1.view(1,-1)
            # sort_data,_=torch.sort(torch.abs(iter_data_1-data0),descending=True)
            # length=sort_data.shape[0]
            # max_j=0
            # for j in range(length):
            #     if torch.sum(sort_data[:j])-(1/(alpha*step_t)+(j+1-1))*sort_data[j]<0:
            #         max_j=j
            #     else:
            #         break
            # belta=(1/(alpha*step_t*(max_j+1)))*torch.sum(sort_data[:max_j+1])
                
            # iter_data_2=iter_data_1 - F.relu(torch.abs(iter_data_1-data0)-belta)*torch.sign(iter_data_1-data0)
            # iter_data=iter_data_2.clone().detach().requires_grad_(True)
    else:
        iter_data=data0.clone().detach().requires_grad_(True)

        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()
            iter_data_1=iter_data+iter_data.grad/torch.norm(iter_data.grad)*step_t
            gap=iter_data_1-iter_data
            # s_,_=torch.max(gap.view(gap.shape[0],-1),1)
            # s_=s_.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,gap.shape[2],gap.shape[3])
            # iter_data_1=torch.clamp(iter_data_1+step_t*s_*(gap==s_)*alpha, min=0, max=1)
            iter_data_1=torch.clamp(iter_data_1-step_t*gap*alpha, min=0, max=1)
            net.zero_grad()
            
            
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
        net.zero_grad()
        
    return iter_data

def prox_z_maxizer_inf(data0,label,step_t,net,max_step,Loss,opt,alpha,is_vector):
    opt.zero_grad()
    if is_vector:
        iter_data=data0.clone().detach().requires_grad_(True)
        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()
            gap=torch.abs(iter_data-data0)
            s_,_=torch.max(gap,1)
            s_=s_.unsqueeze(1).repeat(1,gap.shape[1])
            iter_data_1=iter_data+iter_data.grad*step_t-step_t*s_*(gap==s_)*alpha
            opt.zero_grad()
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
            
            # iter_data_1.view(1,-1)
            # sort_data,_=torch.sort(torch.abs(iter_data_1-data0),descending=True)
            # length=sort_data.shape[0]
            # max_j=0
            # for j in range(length):
            #     if torch.sum(sort_data[:j])-(1/(alpha*step_t)+(j+1-1))*sort_data[j]<0:
            #         max_j=j
            #     else:
            #         break
            # belta=(1/(alpha*step_t*(max_j+1)))*torch.sum(sort_data[:max_j+1])
                
            # iter_data_2=iter_data_1 - F.relu(torch.abs(iter_data_1-data0)-belta)*torch.sign(iter_data_1-data0)
            # iter_data=iter_data_2.clone().detach().requires_grad_(True)
    else:
        iter_data=data0.clone().detach().requires_grad_(True)

        for i in range(max_step):
            iter_output=net(iter_data)
            loss=Loss(iter_output,label)
            loss.backward()


            iter_data_1=torch.clamp(iter_data+iter_data.grad*step_t, min=0, max=1)
            
            
            
            wd_loss=my_loss(iter_data_1,iter_data)*0.02
            wd_loss.backward()
            opt.step()
            opt.zero_grad()
            
            
            
            iter_data=iter_data_1.clone().detach().requires_grad_(True)
        net.zero_grad()
        opt.zero_grad()
        
    return iter_data
        

def turn_batch2one(data,label,step_t,net,max_step,Loss,opt,alpha):
    size=data.shape[0]
    for i in range(size):
        data[i,:]=prox_z_maxizer(data[i,:],label[i],step_t,net,max_step,Loss,opt,alpha,False)
    return data

def turn_batch2one_point(data,label,step_t,net,max_step,Loss,opt,alpha):
    size=data.shape[0]
    for i in range(size):
        data[i,:]=prox_z_maxizer(data[i,:],label[i],step_t,net,max_step,Loss,opt,alpha,True)
    return data


def turn_batch2one_point_inf(data,label,step_t,net,max_step,Loss,opt,alpha):
    size=data.shape[0]

    data=prox_z_maxizer_inf(data,label,step_t,net,max_step,Loss,opt,alpha,True)
    return data

def turn_batch2one_inf(data,label,step_t,net,max_step,Loss,opt,alpha):

    data=prox_z_maxizer_inf3(data,label,step_t,net,max_step,Loss,opt,alpha,False)
    return data
    

def wrm(data,label,step_t,net,max_step,Loss,opt,alpha):
    iter_data=data.clone().detach().requires_grad_(True)
    iter_output=net(iter_data)
    loss=Loss(iter_output,label)
    loss.backward()
    iter_data=(iter_data+iter_data.grad*step_t).clone().detach().requires_grad_(True)
    #iter_data.grad.zero_()
    net.zero_grad()
    for i in range(max_step):
        iter_output=net(iter_data)
        loss=Loss(iter_output,label)
        loss.backward()
        #iter_data_1=iter_data+iter_data.grad/torch.norm(iter_data.grad)*step_t
        grad1=iter_data.grad*step_t
        iter_data.grad.zero_()
        L2_loss=loss_l2(iter_data,data)
        grad2=iter_data.grad
        grad=grad1-grad2
        #iter_data=torch.clamp(iter_data+grad*1/np.sqrt(i+2), min=0, max=1)
        iter_data=(iter_data+grad*1/np.sqrt(i+2)).clone().detach().requires_grad_(True)
        
        
        net.zero_grad()
        
        

    return iter_data
    
        
       

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)



def mynorm(x, order): 
    """
    Custom norm, given x is 2D tensor [b, d]. always calculate norm on the dim=1  
        L1(x) = 1/d * sum(abs(x_i))
        L2(x) = sqrt(1/d * sum(square(x)))
        Linf(x) = max(abs(x_i))
    """
    x = torch.reshape(x, [x.shape[0], -1])
    b, d = x.shape 
    if order == 1: 
        return 1./d * torch.sum(torch.abs(x), dim=1) # [b,]
    elif order == 2: 
        return torch.sqrt(1./d * torch.sum(torch.square(x), dim=1)) # [b,]
    elif order == np.inf:
        return torch.max(torch.abs(x), dim=1)[0] # [b,]
    else: 
        raise ValueError


def uni_wrm(args,X,y,step_t,model,attack_iters,Loss,opt,alpha,restarts=1,norm="l_inf",epsilon=0.03,early_stop=False,mixup=False,lamda=None):
    
    
    
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            #output = model(normalize(X + delta))
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            # if not isinstance(index, slice) and len(index) == 0:
            #     break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                #loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g) # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            tau = alpha # Simply choose tau = alpha 

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)-lamda.detach() * alpha  * (d - torch.sign(d) * epsilon) * (abs_d <= epsilon)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            #all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta+X,max_delta
    
    
def uni_wrm2(args,X,y,model,attack_iters,alpha,tau,restarts=1,norm="l_inf",epsilon=0.03,early_stop=False,mixup=False,lamda=None):
    
    
    
    max_loss = torch.zeros(y.shape[0]).to(args.device)
    max_delta = torch.zeros_like(X).to(args.device)
    early_stop=args.early_stop
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(args.device)
        if norm == "l_inf":
            delta.uniform_(-args.random_init, args.random_init)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, args.clmin-X, args.clmax-X)
        delta.requires_grad = True



        for _ in range(attack_iters):
            #output = model(normalize(X + delta))



            output = model(X + delta)

            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)

            # if not isinstance(index, slice) and len(index) == 0:
            #     break

            if mixup:
                criterion = nn.CrossEntropyLoss()
                #loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g) # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            #tau = alpha # Simply choose tau = alpha 

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()
            
            #max_,_=torch.max(abs_d.view(abs_d.shape[0],-1),axis=1)
            
            
            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)

            d = clamp(d, args.clmin-x, args.clmax-x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            #all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta+X,max_delta


def uni_wrm3(args, X, y, model, attack_iters, alpha, tau, restarts=1, norm="l_inf", epsilon=0.03, early_stop=True,
             mixup=False, lamda=None):
    max_loss = torch.zeros(y.shape[0]).to(args.device)
    max_delta = torch.zeros_like(X).to(args.device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(args.device)
        if norm == "l_inf":
            delta.uniform_(-args.random_init, args.random_init)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, args.clmin - X, args.clmax - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = model(normalize(X + delta))
            output = model(X)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                # loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g)  # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = alpha # Simply choose tau = alpha

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            # max_,_=torch.max(abs_d.view(abs_d.shape[0],-1),axis=1)

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)

            d = clamp(d, args.clmin - x, args.clmax - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            # all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta + X, max_delta


def uni_trades(args, x, y, model, attack_iters, alpha, tau, restarts=1, norm="l_inf", epsilon=0.03, early_stop=False,
             mixup=False, lamda=None):
    max_loss = torch.zeros(y.shape[0]).to(args.device)
    max_delta = torch.zeros_like(x).to(args.device)
    criterion_kl = nn.KLDivLoss(size_average=False)
    for _ in range(restarts):
        delta = torch.zeros_like(x).to(args.device)
        if norm == "l_inf":
            delta.uniform_(-args.random_init, args.random_init)

        delta = clamp(delta, args.clmin - x, args.clmax - x)
        delta.requires_grad = False
        for _ in range(attack_iters):
            output = model(x + delta)

            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)

            if not isinstance(index, slice) and len(index) == 0:
                break

            delta.requires_grad = False
            x_adv=(x+delta).detach().requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + args.epsilon / args.attack_iters * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - args.epsilon), x + args.epsilon)
            x_adv = torch.clamp(x_adv, args.clmin, args.clmax)
            delta.data=x_adv-x
            delta.requires_grad=True


            outputs=model(x)
            loss_nat = F.cross_entropy(outputs, y)
            loss_robust = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(model(x+delta), dim=1),
                                                                 F.softmax(model(x), dim=1))
            loss_main = loss_nat + args.beta * loss_robust


            loss_main.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x_ = x[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g)  # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = alpha # Simply choose tau = alpha

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            # max_,_=torch.max(abs_d.view(abs_d.shape[0],-1),axis=1)

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * args.epsilon) * (abs_d > args.epsilon)

            d = clamp(d, args.clmin - x_, args.clmax - x_)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()


        all_loss = F.cross_entropy(model(x + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta + x, max_delta



def uni_mart(args, x, y, model, attack_iters, alpha, tau, restarts=1, norm="l_inf", epsilon=0.03, early_stop=False,
             mixup=False, lamda=None):
    max_loss = torch.zeros(y.shape[0]).to(args.device)
    max_delta = torch.zeros_like(x).to(args.device)
    criterion_kl = nn.KLDivLoss(size_average=False)
    kl = nn.KLDivLoss(reduction='none')
    for _ in range(restarts):
        delta = torch.zeros_like(x).to(args.device)
        if norm == "l_inf":
            delta.uniform_(-args.random_init, args.random_init)

        delta = clamp(delta,  args.clmin - x, args.clmax - x)
        delta.requires_grad = False
        for _ in range(attack_iters):
            # output = model(x + delta)
            #
            # if early_stop:
            #     index = torch.where(output.max(1)[1] == y)[0]
            # else:
            #     index = slice(None, None, None)
            #
            # if not isinstance(index, slice) and len(index) == 0:
            #     break
            index = slice(None, None, None)
            delta.requires_grad = False
            x_adv=(x+delta).detach().requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, lower_limit, upper_limit)
            delta.data=x_adv-x
            delta.requires_grad=True

            logits = model(x)

            logits_adv = model(x+delta)

            adv_probs = F.softmax(logits_adv, dim=1)

            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

            loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

            nat_probs = F.softmax(logits, dim=1)

            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

            loss_robust = (1.0 / x.shape[0]) * torch.sum(
                torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
            loss = loss_adv + 6.0 * loss_robust





            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x_ = x[index, :, :, :]

            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g)  # equal x_adv = x_adv + alpha * torch.sign(g)

            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            # tau = alpha # Simply choose tau = alpha

            abs_d = torch.abs(d)
            abs_d = abs_d.detach()

            # max_,_=torch.max(abs_d.view(abs_d.shape[0],-1),axis=1)

            d = d - lamda.detach() * alpha / tau * (d - torch.sign(d) * epsilon) * (abs_d > epsilon)

            d = clamp(d, lower_limit - x_, upper_limit - x_)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()


        all_loss = F.cross_entropy(model(x + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta + x, max_delta
        
        

# net=Classifier()
# data=torch.tensor([0.1,2],dtype=torch.float32)
# label=torch.tensor(1)
# Loss=define_loss()
# opt=define_optimizer()
# prox_z_maxizer(data,label,0.1,net,1,Loss,opt,0.1)