# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:19:22 2023

@author: PC
"""


# import pylab as plt
# import numpy as np

# X  = np.linspace(0,3,200)
# Y1 = X**2 + 3
# Y2 = np.exp(X) + 2
# Y4 = Y1+5
# Y3 = np.cos(X)

# plt.plot(X,Y1,lw=4)
# plt.plot(X,Y2,lw=4)
# plt.plot(X,Y3,lw=4)

# plt.fill_between(X, Y1,Y2,color='r',alpha=.2)
# plt.fill_between(X, Y4,Y3,color='b',alpha=.2)

# plt.show()


# import torch

# KL_criterion = torch.nn.KLDivLoss(reduction='none',size_average=False)
# a = torch.tensor([0.1, 0.2, 0.3, 0.4])
# b = torch.tensor([0.1, 0.2, 0.3, 0.4])


import torch.nn as nn
import torch
import torch.nn.functional as F

x = torch.randn((8, 2))
y = torch.randn((8, 2))
# 先转化为概率，之后取对数
x_log = F.log_softmax(x,dim=1)
# 只转化为概率
y = F.softmax(y,dim=1)
kl = nn.KLDivLoss(reduction='none')



print(kl(x_log,y))