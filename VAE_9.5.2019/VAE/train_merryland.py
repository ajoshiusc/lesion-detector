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
import math
import datetime
import os
from keras.datasets import mnist
from torchsummary import summary
import matplotlib.pyplot as plt
from scipy import stats

import VAE_models_functional
from sklearn.model_selection import train_test_split
seed = 10009
epochs =200
batch_size = 8
log_interval = 10
beta=0.00035
sigma=1
z=32

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


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#print(y_test[1:10])
d=np.load('/big_disk/akrami/Projects/lesion_detector_data/VAE_GAN/data_119_maryland.npz')
X=d['data']


max_val=np.max(X)
max_val=np.max(max_val)
#max_val=np.reshape(max_val,(-1,1,1,3))
X = X/ max_val

X = X.astype('float64')
D=X.shape[1]*X.shape[2]*X.shape[3]


X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10002,shuffle=False)
#fig, ax = plt.subplots()
#im = ax.imshow(X_train[0,:,:,0])
print(np.mean(X_train[0,:,:,0]))
#plt.show()
X_train = np.transpose(X_train, (0, 3, 1,2))
X_valid = np.transpose(X_valid , (0, 3, 1,2))

input = torch.from_numpy(X_train).float()
input = input.to('cuda') if args.cuda else input.to('cpu')

validation_data = torch.from_numpy(X_valid).float()
validation_data = validation_data.to('cuda') if args.cuda else validation_data.to('cpu')


train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=True)




model = VAE_models_functional.VAE_nf(z)
model.have_cuda = args.cuda

if args.cuda:
    model.cuda()

print(model)
summary(model, (3, 128, 128))
optimizer = torch.optim.Adam(model.parameters(), lr=3*1e-4)

# Reconstruction + KL divergence losses summed over all elements and batch
def MSE_loss(Y, X):
    ret = (X- Y) ** 2
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,D):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*D/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*3), x.view(-1, 128*128*3), beta,sigma,D)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 128*128*3),x.view(-1, 128*128*3)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = beta_loss_function(recon_batch, data, mu, logvar,beta)
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
            f_data=data[:,2,:,:]
            f_recon_batch=recon_batch[:,2,:,:]
            n = min(f_data.size(0), 100)
            comparison = torch.cat([
            f_data.view(batch_size, 1, 128, 128)[:n],
            f_recon_batch.view(batch_size, 1, 128, 128)[:n],
            (f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
            torch.abs(f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n])
                ])
            save_image(comparison.cpu(),
                           'results/reconstruction_train_' + str(epoch) + '.png',
                           nrow=n)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    return (train_loss / len(train_loader.dataset))


def Validation(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            #        data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
            data = (data).to(device)

            recon_batch, mu, logvar = model(data)
            #print(mu.shape)
            test_loss += beta_loss_function(recon_batch, data, mu,
                                            logvar,beta).item()
            if i == 0:
                f_data=data[:,2,:,:]
                f_recon_batch=recon_batch[:,2,:,:]
                n = min(f_data.size(0), 100)
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    f_recon_batch.view(batch_size, 1, 128, 128)[:n],
                    (f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n]),
                    torch.abs(f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n])
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
    return logvar_all,mu_all,test_loss



if __name__ == "__main__":
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    patience = 10
    no_improvement = 0
    improvment=0
    delta = 0.0001
    for epoch in range(1, epochs + 1):
        train_loss =train(epoch)
        logvar_all,mu_all,validation_loss = Validation(epoch)
        train_loss_list.append(train_loss)
        valid_loss_list.append(validation_loss)

        if validation_loss > best_loss + delta:
            no_improvement += 1
        else:
            no_improvement=0

        best_loss = min(best_loss, validation_loss)


        if no_improvement == patience:
            print("Quitting training for early stopping at epoch ", epoch)
            break
    

    torch.save(model.state_dict(), '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/VAE/results/model_skip_%f_%f_%f.pt' % (z,beta,sigma))

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="validation loss")
    plt.legend()
    plt.show()