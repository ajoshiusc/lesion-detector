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

import VAE_models
from sklearn.model_selection import train_test_split
seed = 10009
epochs = 100
batch_size = 100
log_interval = 10
beta=0.00005
sigma=0.2


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
d=np.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE/data_100_maryland_128.npz')
X=d['data']
#X = X/ 255
X = X.astype('float64')
D=X.shape[1]*X.shape[2]


X_train, X_valid = train_test_split(X, test_size=0.2, random_state=10003)
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


#####test data####
#x_test=np.load('x_test.npy')
#x_test = x_test/ 255
#x_test = x_test.astype('float64')
#y_test=y_test.astype('float64')
#x_test=x_test.reshape((x_test.shape[0],1,x_test.shape[1],x_test.shape[2] ))

#test_data = torch.from_numpy(x_test).float()
#test_data = test_data.to('cuda') if args.cuda else test_data.to('cpu')

#test_loader = torch.utils.data.DataLoader(test_data,
                                          #batch_size=batch_size,
                                         #shuffle=False)

model = VAE_models.AE_nf(256)
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





def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data = (data).to(device)

        optimizer.zero_grad()
        recon_batch, mu = model(data)
        loss = MSE_loss(recon_batch, data)
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
            mu_all=mu
            data_all=data
        else:
            mu_all=torch.cat([
                    mu_all,
                    mu
                ])
            data_all=torch.cat([
                    data_all,
                    data
                ])

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return data_all,mu_all


def Validation(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            data = (data).to(device)

            recon_batch, mu = model(data)
            #print(mu.shape)
            test_loss += MSE_loss(recon_batch, data).item()
            if i == 0:
                f_data=data[:,2,:,:]
                f_recon_batch=recon_batch[:,2,:,:]
                n = min(f_data.size(0), 100)
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 128, 128)[:n],
                    (f_data.view(batch_size, 1, 128, 128)[:n]-f_recon_batch.view(batch_size, 1, 128, 128)[:n])
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png',
                           nrow=n)
               if i == 0:
                mu_all=mu
                data_all=data
            else:
                mu_all=torch.cat([
                    mu_all,
                    mu
                ])
                data_all=torch.cat([
                    data_all,
                    data
                ])
 
    test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return data_all,mu_all


def l21shrink(epsilon, x):
    """
    auther : Chong Zhou
    date : 10/20/2016
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter
        x: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output
    
def L12(Out_teration,inner_iteration)
    teration=20
    inner_iteration=50
    for outer in range(1,Out_iteration+1)
        for epoch in range(1, inner_iteration + 1):
            train(epoch)
            data_all,mu_all=Validation(epoch)
        S=l21shrink(lambda_, (data_all - mu_all))
        input = data_all-S
        train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
    image_X = Image.fromarray(tile_raster_images(X = data_all[:,:,:,2], img_shape = (128,128), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_X.save(r"X.png")
    image_S = Image.fromarray(tile_raster_images(S = data_all[:,:,:,2], img_shape = (128,128), tile_shape=(10, 10),tile_spacing=(1, 1)))
    image_S.save(r"S.png")


if __name__ == "__main__":
    inner = 50
    outer = 20
    lambda_list = [0.0001, 0.0003, 0.0008, 0.001, 0.0015, 0.00035, 0.00045, 
         0.00055, 0.00065, 0.00075, 0.00085, 0.00095, 0.00105, 0.00115, 0.00125]
         print("start")
    for lam in lambda_list:
        folder = "lam" + str(lam)
        folder=results
        L12(Out_teration,inner_iteration)

