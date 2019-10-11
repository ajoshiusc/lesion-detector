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
from torchvision.utils import make_grid, save_image
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import scipy.signal
from VAE_model_pixel64 import Encoder, Decoder, VAE_Generator
import scipy.stats as st
from sklearn import metrics
from sklearn.model_selection import train_test_split

pret = 0

#p_values = scipy.stats.norm.sf(abs(z_scores))*2


def load_model(epoch, encoder, decoder, loc):
    #  restore models
    decoder.load_state_dict(torch.load(loc +
                                       '/VAE_GAN_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc +
                                       '/VAE_GAN_encoder_%d.pth' % epoch))
    encoder.cuda()


#####read data######################
d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/Brats2015_HGG.npz'
)
X = d['data'][:, ::2, ::2, :]

X_data = X[:, :, :, 0:3]
X_data = X_data.astype('float64')

X_train, X_valid = train_test_split(X_data,
                                    test_size=0.2,
                                    random_state=10002,
                                    shuffle=False)
X_valid, X_test = train_test_split(X_valid,
                                   test_size=0.5,
                                   random_state=10001,
                                   shuffle=False)

X_train_seg, X_valid_seg = train_test_split(X,
                                            test_size=0.2,
                                            random_state=10002,
                                            shuffle=False)
X_valid_seg, X_test_seg = train_test_split(X_valid_seg,
                                           test_size=0.5,
                                           random_state=10001,
                                           shuffle=False)

X_train = np.transpose(X_train, (0, 3, 1, 2))
X_valid = np.transpose(X_valid, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

validation_data = torch.from_numpy(X_test).float()

batch_size = 8

torch.manual_seed(7)

Validation_loader = torch.utils.data.DataLoader(validation_data,
                                                batch_size=batch_size,
                                                shuffle=False)
############################################

input_channels = 3
hidden_size = 32
max_epochs = 20
lr = 3e-4
beta = 1
device = 'cuda'
#########################################
epoch = 19
LM = '/home/ajoshi/coding_ground/lesion-detector/VAE_9.5.2019'

##########load low res net##########
G = VAE_Generator(input_channels, hidden_size).cuda()
load_model(epoch, G.encoder, G.decoder, LM)


##########define prob loss##########
def prob_loss_function(recon_x, var_x, x, mu, logvar):
    # x = batch_sz x channel x dim1 x dim2
    x_temp = x.repeat(10, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()

    std = var_x.mul(0.5).exp_()
    #std_all=torch.prod(std,dim=1)
    const = (-torch.sum(var_x * msk, (1, 2, 3))) / 2
    #const=const.repeat(10,1,1,1) ##check if it is correct

    term1 = torch.sum((((recon_x - x_temp) * msk / std)**2), (1, 2, 3))
    const2 = -((64 * 64 * 3) / 2) * math.log((2 * math.pi))

    #term2=torch.log(const+0.0000000000001)
    prob_term = const + (-(0.5) * term1) + const2

    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE + KLD


####################################
def gamma_prob_loss_function(recon_x, var_x, x, mu, logvar, beta):
    x_temp = x.repeat(10, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()

    std = var_x.mul(0.5).exp_()
    #std_all=torch.prod(std,dim=1)
    const = (-torch.sum(var_x*msk, (1, 2, 3))) / 2
    #const=const.repeat(10,1,1,1) ##check if it is correct


    term1 = torch.sum((((recon_x - x_temp)*msk / std)**2), (1, 2, 3))
    const2 = -((128 * 128 * 3) / 2) * math.log((2 * math.pi))

    #term2=torch.log(const+0.0000000000001)
    term2=(beta/beta+1)*torch.sum(var_x.mul(0.5),(1, 2, 3))
    term3=(1/(beta+1))*0.5*(128 * 128 * 3)*math.log(((2 * math.pi)**(beta))*(beta+1))
    prob_term = const + (-(0.5) * term1) + const2+term2+term3

    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE + KLD

def beta_prob_loss_function(recon_x, logvar_x, x, mu, logvar, beta):
    x_temp = x.repeat(10, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-6).float()

    std = logvar_x.mul(0.5).exp_() * msk + 1e-16
    beta = torch.tensor(beta).cuda()
    log_std_all_beta = (torch.log(std) * msk * beta).sum()

    log_term1 = torch.log(1.0 + beta) - torch.log(beta) - (
        beta / 2) * torch.log(torch.tensor(2 * math.pi)) - log_std_all_beta

    term2 = torch.sum(((msk * (recon_x - x_temp) / std)**2), (1, 2, 3))
    logterm2 = -(0.5 * beta * term2)

    term1 = (log_term1 + logterm2).exp()

    term2 = 1 / (log_std_all_beta.exp() * (((beta + 1) *
                                            ((2 * math.pi)**beta))**0.5))

    prob_term = -term1 + term2 + (beta + 1) / beta
    BBCE = torch.sum(prob_term / 10)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BBCE + KLD


##########TEST##########
def Validation(X):
    G.eval()

    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader):
            data = (data).to(device)
            seg = X_test_seg[ind:ind + batch_size, :, :, 3:4]
            msk = torch.tensor(data > 1e-6).float()
            ind = ind + batch_size
            seg = seg.astype('float32')
            seg = torch.from_numpy(seg)

            seg = (seg).to(device)
            mean, logvar, rec_enc, var_enc = G(data)

            seg[seg != 0] = 1

            tem_rec_enc = rec_enc.view(10, -1, 3, 64, 64)
            tem_var_enc = var_enc.view(10, -1, 3, 64, 64)
            std2 = tem_var_enc.exp_()
            mu_all = torch.mean(tem_rec_enc, (0))
            mu2_all = torch.mean((tem_rec_enc**2), (0))
            std2 = torch.mean(std2, (0))

            std_all = std2  #+mu2_all-((mu_all)**2)

            f_recon_batch = mu_all[:, 2, :, :]
            f_data = data[:, 2, :, :]
            sig_plot = ((std_all**(0.5))[:, 2, :, :])
            z_value = (
                (f_data - f_recon_batch) / sig_plot + 1e-16) * msk[:, 2, :, :]
            sig_plot = sig_plot * msk[:, 2, :, :]
            f_recon_batch = f_recon_batch * msk[:, 2, :, :]

            if i < 20:
                n = min(f_data.size(0), 100)
                err_rec = (z_value.view(batch_size, 1, 64, 64)[:n])

                ##########median filtering#############
                median = (err_rec).to('cpu')
                median = median.numpy()
                median = 1 - st.norm.sf(abs(median)) * 2

                median = scipy.signal.medfilt(median, (1, 1, 7, 7))
                scale = 0.05 
                median[median < 1 - scale] = 0
                #median=1-median
                median = median.astype('float32')
                err_rec = torch.from_numpy(median)
                err_rec = (err_rec).to(device)
                ############save_images##############
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    (f_data.view(batch_size, 1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    sig_plot.view(batch_size, 1, 64, 64)[:n] * 10,
                    err_rec.view(batch_size, 1, 64, 64)[:n],
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'reconstruction_b' + str(i) + '.png',
                           nrow=n)
        #############save z values###############
            if i == 0:
                rec_error_all = z_value + 0
            else:
                rec_error_all = torch.cat([rec_error_all, z_value])

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X_test_seg[:, :, :, 3]
    y_true[y_true != 0] = 1

    y_true = np.reshape(y_true, (-1, 1))
    y_true = y_true.astype(int)
    maskX = np.reshape(X_test_seg[:, :, :, 2], (-1, 1))
    y_true = y_true[maskX > 0]

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()

    y_probas = np.reshape(y_probas, (-1, 1, 64, 64))
    median = 1 - ((st.norm.sf(abs(y_probas)) * 2) / (64 * 64))

    #scale=0.05/(64*64)
    median = scipy.signal.medfilt(median, (1, 1, 7, 7))

    y_probas = np.reshape(median, (-1, 1))
    print(np.max(y_true))
    print(np.min(y_true))
    y_probas = y_probas[maskX > 0]

    fpr, tpr, th = metrics.roc_curve(y_true, y_probas)
    L = fpr / tpr
    #best_th=th[tpr>=0.5]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

    y_probas = median + 0
    y_probas[y_probas > 0] = 1

    y_probas = np.reshape(y_probas, (-1, 64 * 64 * 20))
    y_true = np.reshape(y_true, (-1, 64 * 64 * 20))

    dice = 0
    for i in range(y_probas.shape[0]):
        seg = y_probas[i, :]
        gth = y_true[i, :]
        dice += np.sum(seg[gth == 1]) * 2.0 / (np.sum(gth) + np.sum(seg))
        #print((dice))
    print((dice) / y_probas.shape[0])
