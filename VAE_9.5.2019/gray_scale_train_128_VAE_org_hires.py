from __future__ import print_function
import numpy as np
import pywt
from matplotlib import pyplot as plt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
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

pret=0

def show_and_save(file_name,img):
    f = "/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/result_VAE_hi/%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
    #fig = plt.figure(dpi=300)
    #fig.suptitle(file_name, fontsize=14, fontweight='bold')
    #plt.imshow(npimg)
    #plt.imsave(f,npimg)
    
def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), './VAE_GAN_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),'./VAE_GAN_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()
   
    
def load_model(epoch, encoder, decoder):
    #  restore models
    decoder.load_state_dict(torch.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs3/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load('/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs3/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()
  

#####read data######################

X=d['data']
max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))
X = X/ max_val
X = X.astype('float64')
D=X.shape[1]*X.shape[2]
####################################

#########calculate-Wavlet###########
shape = X.shape
max_lev = 1     # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots


fig, axes = plt.subplots(1, 1, figsize=[14, 8])
c = pywt.wavedec2(X, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
#c[0] == tuple([np.zeros_like(v) for v in c[0]])
arr, slices = pywt.coeffs_to_array(c,axes=(1, 2))
arr[:,0:64,0:64,:]=0
##plot##
#axes.imshow(arr[0,:,:,2], cmap=plt.cm.gray)
#axes.set_title('Coefficients\n({} level)'.format(max_lev))
#axes.set_axis_off()
#plt.tight_layout()
#plt.show()
##ormalize##
max_val=np.max(arr)
#arr_final= arr/ np.max(arr)
###################################

##########train validation split##########
batch_size=8
X_train, X_valid = train_test_split(X, test_size=0.1, random_state=10002,shuffle=False)
print(np.mean(X_train[0,:,:,0]))

#axes.imshow(X_valid[0,:,:,2], cmap=plt.cm.gray)
#axes.set_title('Coefficients\n({} level)'.format(max_lev))
#axes.set_axis_off()
#plt.tight_layout()
#plt.show()

c = pywt.wavedec2(X_train, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
#c[0] == tuple([np.zeros_like(v) for v in c[0]])
arr, train_slices = pywt.coeffs_to_array(c,axes=(1, 2))
arr[:,0:64,0:64,:]=0
X_train_ar = np.transpose(arr, (0, 3, 1,2))
X_train_ar=X_train_ar/max_val

c = pywt.wavedec2(X_valid, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
#c[0] == tuple([np.zeros_like(v) for v in c[0]])
arr, valid_slices = pywt.coeffs_to_array(c,axes=(1, 2))
arr[:,0:64,0:64,:]=0
X_valid_ar = np.transpose(arr, (0, 3, 1,2))
X_valid_ar=X_valid_ar/max_val

input = torch.from_numpy(X_train_ar).float()
input = input.to('cuda') 

validation_data = torch.from_numpy(X_valid_ar).float()
validation_data = validation_data.to('cuda') 

X_valid = np.transpose(X_valid, (0, 3, 1,2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference= validation_data_inference.to('cuda') 


train_loader = torch.utils.data.DataLoader(input,
                                           batch_size=batch_size,
                                           shuffle=True)
Validation_loader = torch.utils.data.DataLoader(validation_data,
                                          batch_size=batch_size,
                                          shuffle=False)


Validation_loader_inference = torch.utils.data.DataLoader(validation_data_inference,
                                          batch_size=batch_size,
                                          shuffle=False)
                                         
############################################


##########define network##########
class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, representation_size = 64):
        super(Encoder, self).__init__()
        # input parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.features = nn.Sequential(
            # nc x 128x 128
            nn.Conv2d(self.input_channels, representation_size, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size),
            nn.ReLU(),
            # hidden_size x 64 x 64
            nn.Conv2d(representation_size, representation_size*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 2),
            nn.ReLU(),
            # hidden_size*2 x 32 x 32
            nn.Conv2d(representation_size*2, representation_size*4, 5, stride=2, padding=2),
            nn.BatchNorm2d(representation_size * 4),
            nn.ReLU())
            # hidden_size*4 x 16x 16
            
        self.mean = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
        self.logvar = nn.Sequential(
            nn.Linear(representation_size*4*16*16, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, output_channels))
        
    def forward(self, x):
        batch_size = x.size()[0]

        hidden_representation = self.features(x)

        mean = self.mean(hidden_representation.view(batch_size, -1))
        logvar = self.logvar(hidden_representation.view(batch_size, -1))

        return mean, logvar
    
    def hidden_layer(self, x):
        batch_size = x.size()[0]
        output = self.features(x)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size, representation_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.representation_size = representation_size
        dim = representation_size[0] * representation_size[1] * representation_size[2]
        
        self.preprocess = nn.Sequential(
            nn.Linear(input_size, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        
            # 256 x 16 x 16
        self.deconv1 = nn.ConvTranspose2d(representation_size[0], 256, 5, stride=2, padding=2)
        self.act1 = nn.Sequential(nn.BatchNorm2d(256),
                                  nn.ReLU())
            # 256 x 32 x 32
        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2)
        self.act2 = nn.Sequential(nn.BatchNorm2d(128),
                                  nn.ReLU())
            # 128 x 64 x 64
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, stride=2, padding=2)
        self.act3 = nn.Sequential(nn.BatchNorm2d(32),
                                  nn.ReLU())
            # 32 x 128 x 128
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, stride=1, padding=2)
            # 3 x 128 x 128
        self.activation = nn.Tanh()
            
    
    def forward(self, code):
        bs = code.size()[0]
        preprocessed_codes = self.preprocess(code)
        preprocessed_codes = preprocessed_codes.view(-1,
                                                     self.representation_size[0],
                                                     self.representation_size[1],
                                                     self.representation_size[2])
        output = self.deconv1(preprocessed_codes, output_size=(bs, 256, 32, 32))
        output = self.act1(output)
        output = self.deconv2(output, output_size=(bs, 128, 64, 64))
        output = self.act2(output)
        output = self.deconv3(output, output_size=(bs, 32, 128, 128))
        output = self.act3(output)
        output = self.deconv4(output, output_size=(bs, 3, 128, 128))
        output = self.activation(output)
        return output
class VAE_GAN_Generator(nn.Module):
    def __init__(self, input_channels, hidden_size, representation_size=(256, 16, 16)):
        super(VAE_GAN_Generator, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        
        self.encoder = Encoder(input_channels, hidden_size)
        self.decoder = Decoder(hidden_size, representation_size)
        
    def forward(self, x):
        batch_size = x.size()[0]
        mean, logvar = self.encoder(x)
        std = logvar.mul(0.5).exp_()
        
        reparametrized_noise = Variable(torch.randn((batch_size, self.hidden_size))).cuda()

        reparametrized_noise = mean + std * reparametrized_noise

        rec_images = self.decoder(reparametrized_noise)
        
        return mean, logvar, rec_images


class Discriminator(nn.Module):
    def __init__(self, input_channels, representation_size=(256, 16, 16)):  
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


#################################

########## intilaize parameters##########        
# define constant
input_channels = 3
hidden_size = 64
max_epochs = 500
lr = 3e-4
beta = 0
#########################################

##########call network##########
D = Discriminator(input_channels).cuda()
G = VAE_GAN_Generator(input_channels, hidden_size).cuda()
################################

##########load low res net##########
G2=VAE_GAN_Generator(input_channels, hidden_size).cuda()
load_model(98, G2.encoder, G2.decoder)



##########define optmizer##########
opt_enc = optim.Adam(G.parameters(), lr=lr)
#opt_dec = optim.Adam(G.decoder.parameters(), lr=lr)
##################################

####initialize low information####
fixed_noise = Variable(torch.randn(batch_size, hidden_size)).cuda()
data= next(iter(Validation_loader))
fixed_batch = Variable(data).cuda()

data= next(iter(Validation_loader_inference))
fixed_batch_inference= Variable(data).cuda()
G2.eval()
with torch.no_grad():
    mean,logvar,fake_image_rec=G2(fixed_batch_inference )


data1=(fixed_batch_inference.data).cpu()
data1=data1.numpy()
data1=np.transpose(data1, (0, 2, 3,1))
#axes.imshow(data1[0,:,:,2], cmap=plt.cm.gray)
#axes.set_title('Coefficients\n({} level)'.format(max_lev))
#axes.set_axis_off()
#plt.tight_layout()
#plt.show()

data2=(fake_image_rec.data).cpu()
data2=data2.numpy()
data2=np.transpose(data2, (0, 2, 3,1))


c2 = pywt.wavedec2(data1, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
arr_low, slices = pywt.coeffs_to_array(c2,axes=(1, 2))
#axes.imshow(arr[0,:,:,2], cmap=plt.cm.gray)
#axes.set_title('Coefficients\n({} level)'.format(max_lev))
#axes.set_axis_off()
#plt.tight_layout()
#plt.show()



c3=pywt.wavedec2(data2, 'db2', mode='periodization', level=max_lev,axes=(1, 2))
arr_low_rec, slices = pywt.coeffs_to_array(c3,axes=(1, 2))

##################################


##########define beta loss##########
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

#if pret==1:
    #load_model(499, G.encoder, G.decoder, D)

#####################################


##########train model##########
for epoch in range(max_epochs):
    for data in train_loader:
        batch_size = data.size()[0]
        
        #print (data.size())
        datav = Variable(data).cuda()
        mean, logvar, rec_enc = G(datav)
        # train decoder
        beta_err=beta_loss_function(rec_enc, datav, mean, logvar,beta) 
        err_enc = beta_err
        opt_enc.zero_grad()
        err_enc.backward()
        opt_enc.step()
        
        
    
    print(beta_err)
    _, _, rec_imgs = G(fixed_batch)
    rec_img=rec_imgs.cpu()
    rec_img=rec_img.detach().numpy()
    rec_img_org=np.transpose(rec_img, (0, 2, 3,1))*max_val
    rec_img_org[:,0:64,0:64,:]=arr_low_rec[:,0:64,0:64,:]
    rec_img_org=pywt.array_to_coeffs(rec_img_org,valid_slices,output_format='wavedec2')
    #rec_img_org[0] == c3[0]
    
    rec_img_org=pywt.waverec2(rec_img_org, 'db2', mode='periodization',axes=(1, 2))
    rec_img_org=np.transpose(rec_img_org, (0, 3, 1,2))
    rec_img_org=torch.from_numpy(rec_img_org).float()

    fixed_batch_img=fixed_batch.cpu()
    fixed_batch_img=fixed_batch_img.detach().numpy()
    fixed_batch_img_org=np.transpose(fixed_batch_img, (0, 2, 3,1))*max_val
    fixed_batch_img_org[:,0:64,0:64,:]=arr_low[:,0:64,0:64,:]
    fixed_batch_img_org=pywt.array_to_coeffs(fixed_batch_img_org,valid_slices,output_format='wavedec2')
    #fixed_batch_img_org[0] == c2[0]

    #arr, slices = pywt.coeffs_to_array(fixed_batch_img_org,axes=(1, 2))
    #axes.imshow(arr[0,:,:,2], cmap=plt.cm.gray)
    #axes.set_title('Coefficients\n({} level)'.format(max_lev))
    #axes.set_axis_off()
    #plt.tight_layout()
    #plt.show()

    fixed_batch_img_org=pywt.waverec2(fixed_batch_img_org, 'db2', mode='periodization',axes=(1, 2))
    #axes.imshow(fixed_batch_img_org[0,:,:,2], cmap=plt.cm.gray)
    #axes.set_title('Coefficients\n({} level)'.format(max_lev))
    #axes.set_axis_off()
    #plt.tight_layout()
    #plt.show()
    



    fixed_batch_img_org=np.transpose(fixed_batch_img_org, (0, 3, 1,2))
    fixed_batch_img_org=torch.from_numpy(fixed_batch_img_org).float()


    show_and_save('Input_epoch_wave_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]).cpu(),8))
    show_and_save('rec_epoch_wave_%d.png' % epoch ,make_grid((rec_imgs.data[:,2:3,:,:]).cpu(),8))
    samples = G.decoder(fixed_noise)
    show_and_save('samples_epoch_wave_%d.png' % epoch ,make_grid((samples.data[:,2:3,:,:]).cpu(),8))
    show_and_save('Error_wave_epoch_%d.png' % epoch ,make_grid((fixed_batch.data[:,2:3,:,:]-rec_imgs.data[:,2:3,:,:]).cpu(),8))


    show_and_save('Input_epoch_%d.png' % epoch ,make_grid((fixed_batch_img_org.data[:,2:3,:,:]).cpu(),8))
    show_and_save('rec_epoch_%d.png' % epoch ,make_grid((rec_img_org.data[:,2:3,:,:]).cpu(),8))
    samples = G.decoder(fixed_noise)
    #show_and_save('samples_epoch_%d.png' % epoch ,make_grid((samples.data[:,2:3,:,:]).cpu(),8))
    show_and_save('Error_epoch_%d.png' % epoch ,make_grid((fixed_batch_img_org.data[:,2:3,:,:]-rec_img_org.data[:,2:3,:,:]).cpu(),8))

##########save model###########
save_model(epoch, G.encoder, G.decoder)    
