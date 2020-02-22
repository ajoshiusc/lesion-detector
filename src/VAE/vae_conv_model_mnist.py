from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn


ngf = 64
ndf = 64
nc = 1

class VAE(nn.Module):
    def __init__(self, nz):
        super(VAE, self).__init__()

        self.have_cuda = False
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(128 ),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 32, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.Sigmoid()
            # state size. (ngf*2) x 16 x 16
            
            
            
        )

        self.fc1 = nn.Linear(256*4*4, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 256*4*4)
        

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 256*4*4))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,256,4,4)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

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

class VAE_nf(nn.Module):
    def __init__(self, nz):
        super(VAE_nf, self).__init__()

        self.have_cuda = False
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, 256, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, 128, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(128 ),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 32, 5, 2, 2,1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.Sigmoid()
            # state size. (ngf*2) x 16 x 16
            
            
            
        )

        
        self.cv12 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)
        self.cv22 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)

    

        

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        return self.cv12(conv), self.cv22(conv)

    def decode(self, z):
        return self.decoder(z)

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