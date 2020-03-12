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
from VAE_model_pixel_vanilla_shrink import Encoder, Decoder, VAE_Generator
import copy
import scipy.stats as st

pret=0

def save_model(epoch, encoder, decoder):
    torch.save(decoder.cpu().state_dict(), 'results/VAE_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(), 'results/VAE_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc+'/VAE_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc+'/VAE_encoder_%d.pth' % epoch))
    encoder.cuda()
  


#####read data######################
d=np.load('/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/data_24_ISEL_histeq.npz')
X = d['data']

X_data = X[0:15*20, ::2, ::2, 0:3]
max_val=np.max(X)
#max_val=np.max(max_val,1)
#max_val=np.reshape(max_val,(-1,1,1,3))

X_data = X_data.astype('float64')
X_valid=X_data[:,:,:,:]
D=X_data.shape[1]*X_data.shape[2]
logvar_const=-1.54
var_enc=torch.from_numpy(np.zeros((1))).cuda()+logvar_const
####################################



##########train validation split##########
batch_size=8


X_valid = np.transpose(X_valid, (0, 3, 1,2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference= validation_data_inference.to('cuda') 


Validation_loader_inference = torch.utils.data.DataLoader(validation_data_inference,
                                          batch_size=batch_size,
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
epoch=99
LM='/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/variance shrinkage /results/VAE_vanilla_sigma0.2'

##########load low res net##########
G=VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch,G.encoder, G.decoder,LM)



##########define beta loss##########
def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    dim1=1
    x_temp = x.repeat(dim1, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()
    


    msk2 = torch.tensor(x_temp > -1).float()
    NDim = torch.sum(msk2,(1,2,3))
    std = var_x.mul(0.5).exp_()
    const = (-torch.sum(var_x*msk, (1, 2, 3))) / 2



    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2, 3))
    const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) +const2
    BBCE = torch.sum(prob_term / dim1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   

    return -BBCE + KLD




####################################

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
            seg = X[ind:ind + batch_size, ::2, ::2, 3]
            seg=seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)
            #_, _, arr_lowrec = G(data)
            #f_recon_batch = arr_lowrec[:, 2, :, :]*msk[:, 2, :, :]

            for j in range(100):
                _, _,rec_enc = G(data)
                if j==0:
                    rec_enc_all = copy.deepcopy(rec_enc)
                else:
                    rec_enc_all=torch.cat([rec_enc_all,rec_enc],0)
                         

            mu_all = torch.mean(rec_enc_all.view(100,-1, 3, 64, 64),(0))
            var_all = torch.sum(((rec_enc_all.view(100,-1, 3, 64, 64)-mu_all)**2),(0))/99

            f_recon_batch = mu_all[:, 2, :, :]*msk[:, 2, :, :]
            var_all=var_all[:, 2, :, :]*msk[:, 2, :, :]



            f_data = data[:, 2, :, :]*msk[:, 2, :, :]
            #f_recon_batch = f_recon_batch[:, 2, :, :]
            


            rec_error=((torch.abs(f_data - f_recon_batch))/(1e-16+var_all**(0.5)))*msk[:, 2, :, :]
            sig_plot = ((var_all**(0.5)))
            

            sig_plot = sig_plot * msk[:, 2, :, :]

            #rec_error=torch.mean(rec_error,1)
            if i < 20:
                n = min(f_data.size(0), 100)
                err_rec = (rec_error.view(batch_size, 1, 64, 64)[:n])

                ##########median filtering#############
                median = (err_rec).to('cpu')
                median = median.numpy()
                median = 1 - st.norm.sf(abs(median)) * 2  #Is it one way or two way?

                #median = scipy.signal.medfilt(median, (1, 1, 7, 7))
                scale = 0.05/(64*64) 
                median[median < 1 - scale] = 0
                median = median.astype('float32')
                err_rec = torch.from_numpy(median)
                err_rec = (err_rec).to(device)
                ############save_images##############
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    (f_data.view(batch_size, 1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    sig_plot.view(batch_size, 1, 64, 64)[:n]*5 ,
                    err_rec.view(batch_size, 1, 64, 64)[:n],
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'results/reconstruction_b_dead' + str(i) + '.png',
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
    y_true = X[0:15*20 ,::2, ::2, 3]
    y_true = np.reshape(y_true, (-1, 1))
    
    maskX = np.reshape(X[0:15*20, ::2, ::2, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))
    #y_probas = np.clip(y_probas, 0, 1)
   
    
    y_probas = np.reshape(y_probas, (-1, 1,64,64))
    #y_probas=scipy.signal.medfilt(y_probas,(1,1,7,7))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas[maskX > 0]
    
    fpr, tpr, th= metrics.roc_curve(y_true, y_probas)
    L=fpr/tpr
    best_th=th[fpr<0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()
    
   
    y_probas[y_probas >=best_th[0]]=1
    y_probas[y_probas <best_th[0]]=0

    y_probas = np.reshape(y_probas, (-1,64*64*20))
    y_true = np.reshape(y_true, (-1,64*64*20))
    
    dice=0
    for i in range(y_probas.shape[0]):
        seg=y_probas[i,:]
        gth=y_true[i,:]
        dice += np.sum(seg[gth==1])*2.0 / (np.sum(gth) + np.sum(seg))
        #print((dice))
    print((dice)/y_probas.shape[0])
    
    