# adapted from https://github.com/pytorch/examples/blob/master/vae/main.py
# TODO: modularize
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

seed = 10003
epochs = 100
batch_size = 133
log_interval = 10



torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

beta=0.1

def BBFC_loss(X,Y,beta):
    term1=((beta+1)/beta)
    #print(X)
    #print(Y)
    term2=(X*torch.pow(Y,beta))+(1-X)*torch.pow((1-Y),beta)
    term2=torch.prod(term2, dim=1)
    #print(term2.shape)
    term3=torch.pow(Y,(beta+1))+torch.pow((1-Y),(beta+1))
    term3=torch.prod(term3, dim=1)
    loss1=torch.sum(term1*term2+term3)
    return loss1

# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(recon_x, x, mu, logvar):
    #print(x.shape)
    BBCE = BBFC_loss(recon_x,x,beta)
    #print(recon_x)
    #print(x)
    #print(BBCE)
    #print(recon_x.shape)   
    

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


def loss_function(recon_x, x, mu, logvar):
    #print(x.shape)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #print(recon_x.shape)
    

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = beta_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def test(validation_loader):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            validation_loss += beta_loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    validation_loss /= len(validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(validation_loss))
    return validation_loss


if __name__ == "__main__":
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    patience = 10
    no_improvement = 0
    delta = 0.0001
    for epoch in range(1, epochs + 1):
        train_loss = train(train_loader, epoch)
        validation_loss = test(validation_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(validation_loss)

        if validation_loss > best_loss + delta:
            no_improvement += 1

        best_loss = min(best_loss, validation_loss)

        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            if epoch % 10 == 0:
                save_image(sample.view(64, 1, 28, 28),
                           'results/sample_RVAE_' + str(epoch) + '.png')

        if no_improvement == patience:
            print("Quitting training for early stopping at epoch ", epoch)
            break

    torch.save(model.state_dict(), 'checkpoint.pt')

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="validation loss")
    plt.legend()
    plt.show()