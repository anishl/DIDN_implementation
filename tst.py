#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 18:41:39 2022

@author: anishl
"""
import torch
import torch.nn as nn
import DIDN as d

net = d.DIDN(in_ch=1,N=2,nDub=3,nResRecon=0)
optim = torch.optim.Adam(net.parameters())
loss = nn.MSELoss()    
data = [(torch.rand(5,1,128,128),torch.rand(5,1,128,128)) for i in range(100)]
for i,dat in enumerate(data):
    inp=dat[0]
    tgt=dat[1] 
    optim.zero_grad()
    out = net(inp)
    l = loss(out,tgt)
    l.backward()
    optim.step()
    print(i,l.detach())