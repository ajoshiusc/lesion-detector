from __future__ import print_function
import argparse
import h5py
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import VAE_models_MNIST
import math
from torchsummary import summary
import torchvision

seed = 10009
epochs =500
batch_size = 112
bs=112
log_interval = 10
beta=0
sigma=1
z=32

#####

torch.manual_seed(10001)




def show_and_save(file_name,img):
    f = "./%s.png" % file_name
    save_image(img[:,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder, D):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    torch.save(D.cpu().state_dict(), 'VAE_GAN_D_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
    D.cuda()
    
def load_model(epoch, encoder, decoder, D):
    #  restore models
    decoder.load_state_dict(torch.load('./VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('./VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
    D.load_state_dict(torch.load('VAE_GAN_D_%d.pth' % epoch))
    D.cuda()

#batch_size =16
###
transform=transforms.Compose([
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    
])

transform_anom = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

FRAC_ANOM=0
train_data_gen = datasets.MNIST(root='./data',
                                train=True,
                                transform=transform_anom,
                                download=True)
anom_dataset = datasets.FashionMNIST(root='./mnist_data/',
                               train=True,
                               transform=transform_anom,
                               download=True)

anom_dataset = torch.utils.data.Subset(
    anom_dataset, range(int(FRAC_ANOM * len(train_data_gen))))


train_data_gen = train_data_gen + anom_dataset


valid_data_gen = datasets.MNIST(root='./data',
                                train=False,
                                transform=transform_anom,
                                download=True)
anom_dataset2 = datasets.FashionMNIST(root='./mnist_data/',
                               train=False,
                               transform=transform_anom,
                               download=True)
valid_data_gen = torch.utils.data.Subset(
   valid_data_gen, range(int(112)))
anom_dataset2 = torch.utils.data.Subset(
    anom_dataset2, range(int(FRAC_ANOM * len(valid_data_gen ))))



valid_data_gen=valid_data_gen+anom_dataset2
train_loader = torch.utils.data.DataLoader(dataset=train_data_gen,
                                           batch_size=bs,
                                           shuffle=True,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data_gen,
                                           batch_size=bs,
                                           shuffle=True,
                                           drop_last=True)

# define constant





data= next(iter(valid_loader))
fixed_batch = Variable(data[0]).cuda()
#pret=0
#if pret==1:
    #load_model(499, G.encoder, G.decoder, D)





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
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

ngf = 64
ndf = 64
nc = 1




class encoder(nn.Module):
    def __init__(self, nz):
        super(encoder, self).__init__()

        
        self.nz = nz

        self.encode = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
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

        self.mean=nn.Sequential(
            nn.Linear(256*8*8, 512),
            nn.Linear(512, nz)
        )

        self.logvar=nn.Sequential(
            nn.Linear(256*8*8, 512),
            nn.Linear(512, nz)
        )
            
       
    def forward(self, x):
        conv = self.encode(x)
        mean=self.mean(conv.view(-1, 256*8*8))
        logvar=self.logvar(conv.view(-1, 256*8*8))

        return mean, logvar
    

class decoder(nn.Module):
    def __init__(self,nz):
        super(decoder, self).__init__()
        self.nz = nz
        self.decode = nn.Sequential(
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
        )            

        self.fc3 = nn.Linear(self.nz, 512)
        self.fc4 = nn.Linear(512, 256*8*8)        
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,256,8,8)
        # print("deconv_input", deconv_input.size())
        return self.decode(deconv_input)

class VAE_GAN_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(256, 8, 8)):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = encoder(hidden_size)
        self.decoder = decoder(hidden_size)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        reparametrized_noise=self.reparametrize( mean,logvar)
        rec_images = self.decoder(reparametrized_noise)       
        return mean, logvar, rec_images

def init_normal(m):
    if type(m) == nn.Conv2d:
         nn.init.uniform_(m.weight)
    return()
######
class Discriminator(nn.Module):
    def __init__(self, input_channels, representation_size=(256, 8, 8)):  
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.lth_features = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LeakyReLU(0.2))
        
        self.sigmoid_output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        output = self.sigmoid_output(lth_rep)
        return output
    
    def similarity(self, x):
        batch_size = x.size()[0]
        features = self.main(x)
        lth_rep = self.lth_features(features.view(batch_size, -1))
        return lth_rep

# define constant
input_channels = 1
hidden_size = 64
max_epochs = 500
lr = 1e-3

beta = 0
alpha = 0.1
lamb= 1

model  = VAE_GAN_Generator(input_channels, hidden_size).cuda()
D = Discriminator(input_channels).cuda()

D = Discriminator(input_channels).cuda()

criterion = nn.BCELoss()
criterion.cuda()

opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr)
opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=lr)
opt_dis = optim.Adam(D.parameters(), lr=lr * alpha)
device='cuda'

def train_(epoch):
    model.train()
    train_loss = 0
    D_real_list, D_rec_enc_list, D_rec_noise_list, D_list = [], [], [], []
    g_loss_list, rec_loss_list, prior_loss_list = [], [], []

    for batch_idx, data in enumerate(train_loader):
        #data = (data.gt(0.5).type(torch.FloatTensor)).to(device)
        data=data[0]
        data = (data).to(device)
        ones_label = Variable(torch.ones(batch_size)).cuda()
        zeros_label = Variable(torch.zeros(batch_size)).cuda()
        mu, logvar,recon_batch = model(data)
        noisev = Variable(torch.randn(batch_size, hidden_size)).cuda()
        rec_noise = model.decoder(noisev)
        #train discriminator
        outputD = D(data)
        errD_real = criterion(outputD, ones_label)
        #D_real_list.append(output.data.mean())
        outputD = D(recon_batch)
        errD_rec_enc = criterion(outputD, zeros_label)
        #D_rec_enc_list.append(output.data.mean())
        outputD = D(rec_noise)
        errD_rec_noise = criterion(outputD, zeros_label)
        #D_rec_noise_list.append(output.data.mean())
        
        dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
        #print ("print (dis_img_loss)", dis_img_loss)
        #D_list.append(dis_img_loss.data.mean())
        opt_dis.zero_grad()
        dis_img_loss.backward(retain_graph=True)
        opt_dis.step()
        
        # train decoder
        outputD = D(data)
        errD_real = criterion(outputD, ones_label)
        outputD = D(recon_batch)
        errD_rec_enc = criterion(outputD, zeros_label)
        outputD = D(rec_noise)
        errD_rec_noise = criterion(outputD, zeros_label)
        
        similarity_rec_enc = D.similarity(recon_batch)
        similarity_data = D.similarity(data)
        
        dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
        #print (dis_img_loss)
        gen_img_loss = - dis_img_loss
        
        #g_loss_list.append(gen_img_loss.data.mean())
        rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
        #rec_loss_list.append(rec_loss.data.mean())
        beta_err=beta_loss_function(recon_batch, data, mu, logvar,beta) 
        err_dec =gen_img_loss+rec_loss+beta_err
        
        opt_dec.zero_grad()
        err_dec.backward(retain_graph=True)
        opt_dec.step()

        prior_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        opt_enc.zero_grad()
        loss_enc =  beta_err+rec_loss
        loss_enc.backward(retain_graph=True)
        opt_enc.step()

        #opt_dec.zero_grad()
        #loss_dec = loss_enc
        #loss_dec.backward(retain_graph=True)
        #opt_dec.step()

        
        

        
        train_loss += loss_enc.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_enc.item() / len(data)))
    _,_,rec_imgs = model(fixed_batch)
    show_and_save('Input_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data).cpu(),8))
    show_and_save('rec_epoch_cats_%d.png' % epoch ,make_grid((rec_imgs.data).cpu(),8))
    show_and_save('Error_epoch_cats_%d.png' % epoch ,make_grid((fixed_batch.data-rec_imgs.data).cpu(),8))
         
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len((train_loader.dataset))))
    return (train_loss / len((train_loader.dataset)))   
    
    

  
    



train_loss_list = []
valid_loss_list = []
best_loss = np.inf
for epoch in range(1, epochs + 1):
        train_loss =train_(epoch)

#save_model('cats_%d' % epoch, G.encoder, G.decoder, D)    
    #print (localtime)
    