#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 11:13:59 2022

@author: anishl
"""
import torch.nn as nn
import torch

class _inconv(nn.Module):
    def __init__(self,inch,N):
        super(_inconv, self).__init__()
        self.c1 = nn.Conv2d(inch, N, kernel_size=3,padding=1)
        self.r1 = nn.PReLU()
        self.d1 = nn.Conv2d(N, 2*N, kernel_size=3,stride=2,padding=1)  
        
    def forward(self,x):
        out = self.c1(x)
        out = self.r1(out)
        out = self.d1(out)
        return out
        
class _DUB(nn.Module):
    def __init__(self,inch):
        super(_DUB, self).__init__()
        self.c1 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r1 = nn.PReLU()

        self.c2 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r2 = nn.PReLU()

        self.d3 = nn.Conv2d(inch, 2*inch, kernel_size=3,stride=2,padding=1)

        self.c4 = nn.Conv2d(2*inch, 2*inch, kernel_size=3,padding=1)
        self.r4 = nn.PReLU()

        self.d5 = nn.Conv2d(2*inch, 4*inch, kernel_size=3,stride=2,padding=1)

        self.c6 = nn.Conv2d(4*inch, 4*inch, kernel_size=3,padding=1)
        self.r6 = nn.PReLU()

        self.c7 = nn.Conv2d(4*inch, 8*inch, kernel_size=1,padding=0)

        self.u8 = nn.PixelShuffle(2)

        self.c9 = nn.Conv2d(4*inch, 2*inch, kernel_size=1,padding=0)

        self.c10 = nn.Conv2d(2*inch, 2*inch, 3,padding=1)
        self.r10 = nn.PReLU()

        self.c11 = nn.Conv2d(2*inch, 4*inch, kernel_size=1,padding=0)

        self.u12 = nn.PixelShuffle(2)

        self.c13 = nn.Conv2d(2*inch, inch, kernel_size=1,padding=0)

        self.c14 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r14 = nn.PReLU()

        self.c15 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r15 = nn.PReLU()

        self.c16 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r16 = nn.PReLU()

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.r1(x1)
        x1 = self.c2(x1)
        x1 = self.r2(x1) + x
        # print(x1.shape)

        x2 = self.d3(x1)
        x2 = x2 + self.r4(self.c4(x2))
        # print(x2.shape)
        
        x3 = self.d5(x2)
        x3 = x3+self.r6(self.c6(x3))

        x3 = self.c7(x3)

        x3  = self.u8(x3)
        # print(x3.shape)

        x3 = torch.cat([x2,x3],1)
        x3 = self.c9(x3)
        # print(x3.shape)

        x3 = x3 + self.r10(self.c10(x3))
        x3 = self.c11(x3)
        # print(x3.shape)

        x3 = self.u12(x3)
        # print(x3.shape)
        x3 = torch.cat([x1,x3],1)
        x3 = self.c13(x3)

        x3 = x3 + self.r15(self.c15(self.r14(self.c14(x3))))
        x3 = x + self.r16(self.c16(x3))
        return x3

class _ResBlock(nn.Module):
    def __init__(self,inch):
        super(_ResBlock,self).__init__()
        self.c1 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r1 = nn.PReLU()
        
        self.c2 = nn.Conv2d(inch, inch, kernel_size=3,padding=1)
        self.r2 = nn.PReLU()
        
    def forward(self, x):
        out = x + self.r2(self.c2(self.r1(self.c1(x))))
        return out
        
class _ReconBlock(nn.Sequential):
    def __init__(self,inch,nBlocks):
        super(_ReconBlock,self).__init__()
        for i in range(nBlocks):
            self.add_module('resblock',_ResBlock(inch))
        self.add_module('finconv', nn.Conv2d(inch, inch, 3,padding = 1))
        
        
class DIDN(nn.Module):
    def __init__(self,in_ch,N,nDub,nResRecon):
        assert nDub>0, 'at least one DUB block is necessary'
        super(DIDN, self).__init__()
        self.init1 = _inconv(in_ch, N)
        self.dubs=[]
        for i in range(nDub):
            self.dubs.append(_DUB(2*N))
        self.dubs = nn.ModuleList(self.dubs)
        self.recon = _ReconBlock(2*N,nResRecon)
        
        self.mid2 = nn.Conv2d(2*N*nDub, 4*N, kernel_size=1,padding=0)
        self.c2 = nn.Conv2d(4*N, 4*N, kernel_size=3,padding=1)
        self.r2  =nn.PReLU()
        self.u2  =nn.PixelShuffle(2)
        
        self.finconv = nn.Conv2d(N, in_ch, kernel_size=3,padding=1)
        self.finrel = nn.PReLU()
        self.nDub =nDub
        
    def forward(self,x):
        out1 = self.init1(x)
        out = []
        for i in range(self.nDub):
            out1 = self.dubs[i](out1)
            out1 = self.recon(out1)
            out.append(out1)
        out = torch.cat(out,1)
        out = self.mid2(out)
        out = out + self.r2(self.c2(out))
        out = self.u2(out)
        out = self.finrel(self.finconv(out))+x
        return out
        
        
            
            
            
            
        
        
