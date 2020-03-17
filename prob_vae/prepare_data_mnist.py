from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from keras.datasets import mnist
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from vaemodel import VAE
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
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
                    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=50,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


#(x_train, _), (x_test, _) = mnist.load_data()

(X, _), (_, _) = mnist.load_data()
X = X / 255
X = X.astype(float)
#x_train, x_test = train_test_split(X)

'''x_train = x_train / 255
x_train = x_train.astype(float)
x_test = x_test / 255
x_test = x_test.astype(float)'''

in_data = X #np.concatenate((x_train, x_test), axis=0)
in_data = torch.tensor(in_data).float().view(in_data.shape[0], 1, 28, 28)

#x_train = torch.from_numpy(x_train).float().view(x_train.shape[0],1,28,28)
#x_test = torch.from_numpy(x_test).float().view(x_test.shape[0],1,28,28)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.load_state_dict(torch.load('results/VAE_mean.pth'))

out_data = np.zeros(in_data.shape)

model.eval()
with torch.no_grad():

    for i, data in enumerate(tqdm(in_data)):
        data = data[None, ].to(device)
        rec, mean, logvar = model(data)
        out_data[i, ] = rec.view(1, 28, 28).cpu()

np.savez('results/rec_data_mnist.npz', out_data=out_data, in_data=in_data)
