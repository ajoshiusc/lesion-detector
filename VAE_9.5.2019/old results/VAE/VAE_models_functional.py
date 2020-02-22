from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


ngf = 64
ndf = 64
nc = 3



class VAE_nf(nn.Module):
    def __init__(self, nz):
        super(VAE_nf, self).__init__()
        ###encoder
        #self.have_cuda = False
        self.nz = nz
        self.e1=nn.Conv2d(nc, 64, 5, 2, 2, bias=False)
            # state size. (ndf) x 14 x 14
        self.e2=nn.Conv2d(64, 128, 5, 2, 2, bias=False)
        self.bn1=nn.BatchNorm2d(128)
        self.dr1=nn.Dropout(p=0.5)
            # state size. (ndf*2) x 7 x 7
        self.e3=nn.Conv2d(128, 256, 5, 2, 2, bias=False)
        self.bn2=nn.BatchNorm2d(256)
        self.dr2=nn.Dropout(p=0.5)
        ###decoder    
        self.d1=nn.ConvTranspose2d(self.nz, 256, 5, 2, 2,1, bias=False)
        self.bn3=nn.BatchNorm2d(256)
        self.dr3=nn.Dropout(p=0.5)
            # state size. (ngf*8) x 4 x 4
        self.d2=nn.ConvTranspose2d(512, 256, 5, 2, 2,1, bias=False)
        self.bn4=nn.BatchNorm2d(256)
        self.dr4=nn.Dropout(p=0.5)
            # state size. (ngf*4) x 8 x 8
        self.d3=nn.ConvTranspose2d(384, 128, 5, 2, 2,1, bias=False)
        self.bn5=nn.BatchNorm2d(128 )
        self.d4=nn.ConvTranspose2d(128, 32, 5, 2, 2,1, bias=False)
        self.bn6=nn.BatchNorm2d(32)
        self.d5=nn.Conv2d(32, nc, 5, 1, 2, bias=False)

        
        self.cv12 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)
        self.cv22 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)

    

        

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        self.conv0 = self.relu(self.e1(x))
        self.conv1=self.relu(self.bn1(self.e2(self.conv0)))
        self.conv2=self.dr1(self.conv1)
        self.conv3=self.relu(self.bn2(self.e3(self.conv2)))
        self.conv4=self.dr2(self.conv3)
        return self.cv12(self.conv4), self.cv22(self.conv4)

    def decode(self, z):
        z=self.d1(z)
        z=self.bn3(z)
        z=torch.cat(((self.dr3(self.relu(z))),self.conv4),1)
        z=self.d2(z)
        z=self.bn4(z)
        z=torch.cat(((self.dr4(self.relu(z))),self.conv2),1)
        z=self.d3(z)
        z=self.bn5(self.relu(z))
        z=self.d4(z)
        z=self.bn6(self.relu(z))
        z=self.d5(z)
        

        return self.sigmoid(z)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        # print("decoded", decoded.size())
        return decoded, mu, logvar

