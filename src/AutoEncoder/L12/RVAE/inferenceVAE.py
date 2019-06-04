from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from VAE_pytorch import VAE
seed = 10003
epochs = 100
batch_size = 64
log_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


X = np.load('lesion_x_train.npy')
X_train, X_valid = train_test_split(X, test_size=0.33, random_state=10003)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_valid = X_valid.reshape((len(X_valid), np.prod(X_valid.shape[1:])))


input = torch.from_numpy(X_train).float()
input = input.to('cuda') if torch.cuda.is_available() else input

validation = torch.from_numpy(X_valid).float()
validation = input.to('cuda') if torch.cuda.is_available() else input

train_loader = torch.utils.data.DataLoader(input, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)

model = VAE()
model.to(device)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()


#$model=torch.load('checkpoint2.pth')
# Reconstruction + KL divergence losses summed over all elements and batch
beta=0.1








    #validation_loss = 0
with torch.no_grad():
    for i, data in enumerate(validation_loader):
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        BCE = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
        print(BCE.shape)
        if i == 0:
            save_image(data.view(64, 1, 28, 28),
                        'results/X_' + str(i) + '.png')
            save_image(BCE.view(64, 1, 28, 28),
                        'results/BCE_X_' + str(i) + '.png')
            save_image(recon_batch.view(64, 1, 28, 28),
                        'results/re_X_' + str(i) + '.png')

    #validation_loss /= len(validation_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(validation_loss))


