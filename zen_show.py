# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:12:40 2023

@author: PC
"""

import matplotlib.pyplot as plt

import matplotlib.image as Image
name_list=['zenuni','zentrades','zenunitrade','zenunimart']
n=5
t=4

for k in range(n):
    ax=plt.subplot(t+1,n,k+1)
    plt.tight_layout(h_pad=0)
    im=Image.imread("{}/input_{}.png".format(name_list[0],k))
    plt.imshow(im)
    plt.axis('off')
for i in range(t):
    
    for k in range(n):
        ax=plt.subplot(t+1,n,i*n+k+1+n)
        
        im=Image.imread("{}/symmetric3_{}.png.png".format(name_list[i],k))
        plt.imshow(im)
        plt.axis('off')

plt.savefig("kk.png")
        
        


