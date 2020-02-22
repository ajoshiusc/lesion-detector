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
from sklearn import metrics
import scipy.signal
from sklearn.model_selection import train_test_split
from VAE_model_pixel_vanilla import Encoder, Decoder, VAE_Generator

pret=0

def show_and_save(file_name,img):
    f = "/big_disk/akrami/git_repos/lesion-detector/src/VAE_GANs/figs4/%s.png" % file_name
    save_image(img[2:3,:,:],f)
    
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
    
def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()

#####read data######################

d=np.load('data__maryland_histeq.npz')
X=d['data']
X=X[0:2380,:,:,:]
X_train=X[0:-20*20,:,:,:]
X_valid=X[-20*20:,:,:,:]


d=np.load('data__TBI_histeq.npz')
X_train=np.rot90(np.concatenate((X_train,d['data'][0:-20*20,:,:,:]),axis=0), k=3, axes=(1, 2))
X_valid=np.rot90(np.concatenate((X_valid,d['data'][-20*20:,:,:,:]),axis=0),k=3, axes=(1, 2))


d=np.load('Brats2015_HGG.npz')
X_data=d['data']

X_test_seg=np.rot90(X_data[0:20*20,:,:,:], k=2, axes=(1, 2))
X_test = np.transpose(X_test_seg[:,::2,::2,0:3] , (0, 3, 1,2))

X_test = torch.from_numpy(X_test.copy()).float()

Validation_loader_inference = torch.utils.data.DataLoader(X_test,
                                          batch_size=8,
                                          shuffle=False)
############################################



###### define constant########
input_channels = 3
hidden_size =128
max_epochs = 100
lr = 3e-4
beta =0
device='cuda'
#########################################
epoch=39
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/VAE_original_final'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################
batch_size=8
##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X_test_seg[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<20:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size,1, 64, 64)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    err.view(batch_size, 1, 64, 64)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 64, 64)[:n] -
                        f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'VAE_original_final/reconstruction_br_vic' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[0:20*20 ,::2, ::2, 3]
    y_true[y_true!=0]=1
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X_test_seg[0:20*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="VAE, auc=" + "{:.2f}".format(auc))
    plt.legend(loc=4)
   
epoch=99
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/VAE_original_final'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################
batch_size=8
##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X_test_seg[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<20:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size,1, 64, 64)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    err.view(batch_size, 1, 64, 64)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 64, 64)[:n] -
                        f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'VAE_original_final/reconstruction_br' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[0:20*20 ,::2, ::2, 3]
    y_true[y_true!=0]=1
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X_test_seg[0:20*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="VAEbr, auc=" + "{:.2f}".format(auc))
    plt.legend(loc=4)


epoch=99
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/RVAE_final'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################
batch_size=8
##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X_test_seg[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<20:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size,1, 64, 64)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    err.view(batch_size, 1, 64, 64)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 64, 64)[:n] -
                        f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'RVAE_final/reconstruction_br_vic' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[0:20*20 ,::2, ::2, 3]
    y_true[y_true!=0]=1
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X_test_seg[0:20*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="RVAEbr, auc=" + "{:.2f}".format(auc))
    plt.legend(loc=4)


epoch=99
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Pre_t_brats'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################
batch_size=8
##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X_test_seg[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<20:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size,1, 64, 64)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    err.view(batch_size, 1, 64, 64)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 64, 64)[:n] -
                        f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'Pre_t_brats/reconstruction_br_vic' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[0:20*20 ,::2, ::2, 3]
    y_true[y_true!=0]=1
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X_test_seg[0:20*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="PreVAE, auc=" + "{:.2f}".format(auc))
    plt.legend(loc=4)
    
    
    
epoch=99
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/RVAE_pre_t_brats'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########

def MSE_loss(Y, X):
    msk = torch.tensor(X > 1e-6).float()
    ret = ((X- Y) ** 2)*msk
    ret = torch.sum(ret,1)
    return ret 
def BMSE_loss(Y, X, beta,sigma,Dim):
    term1 = -((1+beta) / beta)
    K1=1/pow((2*math.pi*( sigma** 2)),(beta*Dim/2))
    term2=MSE_loss(Y, X)
    term3=torch.exp(-(beta/(2*( sigma** 2)))*term2)
    loss1=torch.sum(term1*(K1*term3-1))
    return loss1



# Reconstruction + KL divergence losses summed over all elements and batch

def beta_loss_function(recon_x, x, mu, logvar, beta):

    if beta > 0:
        sigma=1
        # If beta is nonzero, use the beta entropy
        BBCE = BMSE_loss(recon_x.view(-1, 128*128*1), x.view(-1, 128*128*1), beta,sigma,128*128*1)
    else:
        # if beta is zero use binary cross entropy
        BBCE = torch.sum(MSE_loss(recon_x.view(-1, 64*64*1),x.view(-1, 64*64*1)))

    # compute KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE +KLD

####################################
batch_size=8
##########TEST##########
def Validation(X):
    G.eval()
#G2.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > 1e-6).float()
            seg = X_test_seg[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            _, _, arr_lowrec = G(data)
            f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            

            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            rec_error = torch.abs(f_data - f_recon_batch)*msk[:, 2, :, :]
            #rec_error=torch.mean(rec_error,1)
            if i<20:
                n = min(f_data.size(0), 100)
                err=torch.abs(f_data.view(batch_size,1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size,1, 64, 64)[:n])
                err=err
                #err=torch.mean(err,1)
                median=(err).to('cpu')
                median=median.numpy()
                median=scipy.signal.medfilt(median,(1,1,7,7))
                median=median.astype('float32')
                median = np.clip(median, 0, 1)
    
                err=median
                err=torch.from_numpy(err)
                err=(err).to(device)

                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    err.view(batch_size, 1, 64, 64)[:n],
                    torch.abs(
                        f_data.view(batch_size, 1, 64, 64)[:n] -
                        f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'RVAE_pre_t_brats/reconstruction_br_vic' +str(i)+ '.png',
                           nrow=n)
                
            if i==0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    #test_loss /= len(Validation_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[0:20*20 ,::2, ::2, 3]
    y_true[y_true!=0]=1
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X_test_seg[0:20*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="PreRVAE, auc=" + "{:.2f}".format(auc))
    plt.legend(loc=4)
    plt.show()