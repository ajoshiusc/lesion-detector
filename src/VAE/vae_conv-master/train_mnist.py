from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import json
import numpy as np

import datetime
import os
from keras.datasets import mnist

import vae_conv_model_mnist
from sklearn.model_selection import train_test_split
seed = 10009
epochs = 100
batch_size = 100
log_interval = 10

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='enables CUDA training')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (import matplotlib.pyplot as pltult: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(seed)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#print(y_test[1:10])
X=np.load('X.npy')
x_test=np.load('x_test.npy')
X = X/ 255
X = X.astype('float64')
x_test = x_test/ 255
x_test = x_test.astype('float64')
#y_test=y_test.astype('float64')


X_train, X_valid = train_test_split(X, test_size=0.33, random_state=10003)
X_train = X_train.reshape((X_train.shape[0],1, X_train.shape[1],X_train.shape[2]))
X_valid = X_valid.reshape((X_valid.shape[0],1,X_valid.shape[1],X_valid.shape[2] ))
x_test=x_test.reshape((x_test.shape[0],1,x_test.shape[1],x_test.shape[2] ))
input = torch.from_numpy(X_train).float()
input = input.to('cuda') if args.cuda else input.to('cpu')

validation_data = torch.from_numpy(X_valid).float()
validation_data = validation_data.to('cuda') if args.cuda else validation_data.to('cpu')

test_data = torch.from_numpy(x_test).float()
test_data = test_data.to('cuda') if args.cuda else test_data.to('cpu')

train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

model = vae_conv_model_mnist.VAE_nf(256)
model.have_cuda = args.cuda

if args.cuda:
    model.cuda()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret)
    return ret 

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

    
    BBCE = MSE_loss(recon_x, x)

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
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

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def Validation(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += loss_function(recon_batch, data, mu,
                                            logvar).item()
            if i == 0:
                n = min(data.size(0), 100)
                comparison = torch.cat([
                    data.view(batch_size, 1, 32, 32)[:n],
                    recon_batch.view(batch_size, 1, 32, 32)[:n]
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n)
                mu_all=mu
                logvar_all=logvar
            else:
                mu_all=torch.cat([
                    mu_all,
                    mu
                ])
                logvar_all=torch.cat([
                    logvar_all,
                    logvar
                ])
    test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all,mu_all


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += loss_function(recon_batch, data, mu,
                                            logvar).item()
            if i == 0:
                mu_all=mu
                logvar_all=logvar
            else:
                mu_all=torch.cat([
                    mu_all,
                    mu
                ])
                logvar_all=torch.cat([
                    logvar_all,
                    logvar
                ])
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return logvar_all,mu_all


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        logvar_all,mu_all=Validation(epoch)
        logvar_all_test,mu_all_test=test(epoch)
        mu_all_test=mu_all_test.cpu()
        #mu_all=mu_all.cpu()
        Np_mu=mu_all_test.numpy()