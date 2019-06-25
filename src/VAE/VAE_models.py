from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IM_SZ = 64
ngf = 64
ndf = 64
nc = 3
SIGMA = 1.0
BETA = 0


def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret, 1)
    return ret


def Gaussian_CE_loss(Y, X, beta, sigma=SIGMA):  # 784 for mnist
    D = Y.shape[1]
    term1 = -((1 + beta) / beta)
    K1 = 1 / pow((2 * math.pi * (sigma**2)), (beta * D / 2))
    term2 = MSE_loss(Y, X)
    term3 = torch.exp(-(beta / (2 * (sigma**2))) * term2)
    loss1 = torch.sum(term1 * (K1 * term3 - 1))
    return loss1


def Bernoulli_CE_loss(Y, X, beta):
    term1 = (1 / beta)
    term2 = (X * torch.pow(Y, beta)) + (1 - X) * torch.pow((1 - Y), beta)
    term2 = torch.prod(term2, dim=1) - 1
    term3 = torch.pow(Y, (beta + 1)) + torch.pow((1 - Y), (beta + 1))
    term3 = torch.prod(term3, dim=1) / (beta + 1)
    loss1 = torch.sum((-term1 * term2 + term3) * (beta + 1))
    if torch.isnan(loss1):
        print(loss1)
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def bdiv_elbo(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        #BBCE = Bernoulli_CE_loss(recon_x, x, beta)
        BBCE = Gaussian_CE_loss(recon_x, x, beta)
    else:
        # if beta is zero use binary cross entropy
        #BBCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BBCE = torch.sum(MSE_loss(recon_x, x))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


class VAE(nn.Module):
    def __init__(self, nz):
        super(VAE, self).__init__()

        self.have_cuda = False
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, IM_SZ, 5, 2, 2, bias=False),
            nn.BatchNorm2d(IM_SZ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(IM_SZ, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, IM_SZ, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(IM_SZ),
            nn.ReLU(True),
            nn.ConvTranspose2d(IM_SZ, 32, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.Sigmoid()
            # state size. (ngf*2) x 16 x 16
        )

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 256 * 4 * 4)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 256 * 4 * 4))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1, 256, 4, 4)
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
            nn.Conv2d(nc, 64, 5, 2, 2, bias=True),
            nn.ReLU(True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, IM_SZ, 5, 2, 2, bias=True),
            nn.BatchNorm2d(IM_SZ),
            nn.ReLU(True),
            #  nn.Dropout(p=0.5),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(IM_SZ, 256, 5, 2, 2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.Dropout(p=0.5)
            # state size. (ndf*4) x 4 x 4
            #nn.Conv2d(256, 256, 5, 2, 2, bias=False),
            #nn.BatchNorm2d(256),
            #nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, 256, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #   nn.Dropout(p=0.5),
            # state size. (ngf*8) x 4 x 4
            #nn.ConvTranspose2d(256, 256, 5, 2, 2,1, bias=False),
            #nn.BatchNorm2d(256),
            #nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #   nn.Dropout(p=0.5),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, IM_SZ, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(IM_SZ),
            nn.ReLU(True),
            nn.ConvTranspose2d(IM_SZ, 32, 5, 2, 2, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, nc, 5, 1, 2, bias=True),
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
        #conv=conv.cuda()
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


class AE_nf(nn.Module):
    def __init__(self, nz):
        super(AE_nf, self).__init__()

        self.have_cuda = False
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
            nn.ReLU(True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(64, IM_SZ, 5, 2, 2, bias=False),
            nn.BatchNorm2d(IM_SZ),
            nn.ReLU(True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(IM_SZ, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(256, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, IM_SZ, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(IM_SZ),
            nn.ReLU(True),
            nn.ConvTranspose2d(IM_SZ, 32, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, nc, 5, 1, 2, bias=False),
            nn.Sigmoid()
            # state size. (ngf*2) x 16 x 16
        )

        self.cv12 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)
        #self.cv22 = nn.Conv2d(256, self.nz, 5, 2, 2, bias=False)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        return self.cv12(conv)

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
        z = self.encode(x)
        decoded = self.decode(z)
        # print("decoded", decoded.size())
        return decoded, z


def train(model, data, device='cuda', epochs=10, batch_size=32, patience=100):
    best_loss = np.inf
    no_improvement = 0

    model.train()
    #data = (data).to(device)

    X_train, X_valid = train_test_split(data,
                                        test_size=0.1,
                                        random_state=10002,
                                        shuffle=False)

    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_train = (X_train).to(device)
    X_valid = (X_valid).to(device)

    train_loader = torch.utils.data.DataLoader(X_train,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(X_valid,
                                               batch_size=batch_size,
                                               shuffle=True)

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model,
                                 train_loader,
                                 device=device,
                                 epoch=epoch,
                                 batch_size=batch_size)

        logvar_all, mu_all, validation_loss = valid_epoch(
            model,
            valid_loader,
            device=device,
            epoch=epoch,
            batch_size=batch_size)

        train_loss_list.append(train_loss)
        valid_loss_list.append(validation_loss)

        if validation_loss > best_loss:
            no_improvement += 1
        else:
            no_improvement = 0

        best_loss = min(best_loss, validation_loss)

        if no_improvement == patience:
            print("Quitting training for early stopping at epoch ", epoch)
            break
    
    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="validation loss")
    plt.legend()
    plt.show()


def train_epoch(model,
                train_loader,
                epoch=0,
                device='cuda',
                batch_size=32,
                log_interval=10):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = bdiv_elbo(
            recon_batch, data, mu, logvar, beta=BETA
        )  # beta_loss_function(recon_batch, data, mu, logvar, beta)
        loss.backward()
        if torch.isnan(loss):
            print(loss)
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        if batch_idx == 0:
            f_data = data[:, 2, :, :]
            f_recon_batch = recon_batch[:, 2, :, :]
            n = min(f_data.size(0), 100)
            comparison = torch.cat([
                f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n],
                f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n],
                (f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n] -
                 f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n]),
                torch.abs(
                    f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n] -
                    f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n])
            ])
            save_image(comparison.cpu(),
                       'results/reconstruction_train_' + str(epoch) + '.png',
                       nrow=n)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    return (train_loss / len(train_loader.dataset))


def valid_epoch(model,
                valid_loader,
                epoch=0,
                device='cuda',
                batch_size=32,
                log_interval=10):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += bdiv_elbo(recon_batch, data, mu, logvar,
                                   beta=BETA).item()
            if i == 0:
                f_data = data[:, 2, :, :]
                f_recon_batch = recon_batch[:, 2, :, :]
                n = min(f_data.size(0), 100)
                comparison = torch.cat([
                    f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n],
                    f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n],
                    (f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n] -
                     f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n]),
                    torch.abs(
                        f_data.view(batch_size, 1, IM_SZ, IM_SZ)[:n] -
                        f_recon_batch.view(batch_size, 1, IM_SZ, IM_SZ)[:n])
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n)
                mu_all = mu
                logvar_all = logvar
            else:
                mu_all = torch.cat([mu_all, mu])
                logvar_all = torch.cat([logvar_all, logvar])
    test_loss /= len(valid_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all, mu_all, test_loss
