"""Plot ROCs.

Created on Wed Mar  4 09:44:29 2020

@author: ajoshi
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torchvision.utils import save_image
from sklearn import metrics
from VAE_model import VAE_Generator
import scipy.stats as st
import math

pret = 0


def save_model(epoch, encoder, decoder):
    """Save the model to a file."""
    torch.save(decoder.cpu().state_dict(),
               'results/VAE_decoder_%d.pth' % epoch)
    torch.save(encoder.cpu().state_dict(),
               'results/VAE_encoder_%d.pth' % epoch)
    decoder.cuda()
    encoder.cuda()


def load_model(epoch, encoder, decoder, loc):
    """Load the model from a file for mean."""
    decoder.load_state_dict(torch.load(loc + '/VAE_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc + '/VAE_encoder_%d.pth' % epoch))
    encoder.cuda()


def load_model_std(epoch, encoder, decoder, loc):
    """Load the model from a file for stddev."""
    decoder.load_state_dict(torch.load(loc +
                                       '/VAE_std_decoder_%d.pth' % epoch))
    decoder.cuda()
    encoder.load_state_dict(torch.load(loc +
                                       '/VAE_std_encoder_%d.pth' % epoch))
    encoder.cuda()


d = np.load(
    '/big_disk/akrami/git_repos_new/lesion-detector/VAE_9.5.2019/old results/\
data_24_ISEL_histeq.npz'
)
X = d['data']

X_data = X[0:15 * 200, ::2, ::2, 0:3]
max_val = np.max(X)

X_data = X_data.astype('float64')
X_valid = X_data[:, :, :, :]
D = X_data.shape[1] * X_data.shape[2]
####################################

# train validation split
batch_size = 8

X_valid = np.transpose(X_valid, (0, 3, 1, 2))
validation_data_inference = torch.from_numpy(X_valid).float()
validation_data_inference = validation_data_inference.to('cuda')

Validation_loader_inference = torch.utils.data.DataLoader(
    validation_data_inference, batch_size=batch_size, shuffle=False)

############################################

# define constants
input_channels = 3
hidden_size = 8
max_epochs = 100
lr = 3e-4
beta = 0
device = 'cuda'
#########################################
LM = 'results'

# load low res net
Gmean = VAE_Generator(input_channels, hidden_size).cuda()
load_model(24, Gmean.encoder, Gmean.decoder, LM)

Gstd = VAE_Generator(input_channels, hidden_size).cuda()
load_model_std(77, Gstd.encoder, Gstd.decoder, LM)


def prob_loss_function(recon_x, var_x, x, mu, logvar):
    """Define prob loss function."""
    dim1 = 1
    x_temp = x.repeat(dim1, 1, 1, 1)
    msk = torch.tensor(x_temp > 1e-3).float()

    msk2 = torch.tensor(x_temp > 1e-3).float()
    NDim = torch.sum(msk2, (1, 2, 3))
    std = var_x.mul(0.5).exp_()
    const = (-torch.sum(var_x * msk, (1, 2, 3))) / 2

    term1 = torch.sum((((recon_x - x_temp) * msk / std)**2), (1, 2, 3))
    const2 = -(NDim / 2) * math.log((2 * math.pi))

    prob_term = const + (-(0.5) * term1) + const2
    BBCE = torch.sum(prob_term / dim1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return -BBCE + KLD


def Validation(X):
    """Validation."""
    Gmean.eval()
    test_loss = 0
    ind = 0
    with torch.no_grad():
        for i, data in enumerate(Validation_loader_inference):
            data = (data).to(device)
            msk = torch.tensor(data > -1e6).float()
            seg = X[ind:ind + batch_size, ::2, ::2, 3]
            seg = seg.astype('float32')
            ind = ind + batch_size
            seg = torch.from_numpy(seg)
            seg = (seg).to(device)

            _, _, rec_enc_mean = Gmean(data)
            _, _, rec_enc_std = Gstd(data)

            rec_enc_std[rec_enc_std < 0] = 0

            mu_all = rec_enc_mean
            # torch.mean(rec_enc_all.view(100,-1, 3, 64, 64),(0))
            var_all = rec_enc_std**2

            f_recon_batch = mu_all[:, 2, :, :] * msk[:, 2, :, :]
            var_all = var_all[:, 2, :, :] * msk[:, 2, :, :]

            f_data = data[:, 2, :, :] * msk[:, 2, :, :]

            rec_error = ((torch.abs(f_data - f_recon_batch)) /
                         (1e-6 + var_all**0.5)) * msk[:, 2, :, :]
            sig_plot = ((var_all**(0.5)))

            sig_plot = sig_plot * msk[:, 2, :, :]

            if i < 20:
                n = min(f_data.size(0), 100)
                err_rec = (rec_error.view(batch_size, 1, 64, 64)[:n])

                # median filtering
                median = (err_rec).to('cpu')
                median = median.numpy()
                median = 1 - st.norm.sf(
                    abs(median)) * 2  # Is it one way or two way?

                scale = 0.05 / (64 * 64)
                median[median < 1 - scale] = 0
                median = median.astype('float32')
                err_rec = torch.from_numpy(median)
                err_rec = (err_rec).to(device)
                # save_images
                comparison = torch.cat([
                    f_data.view(batch_size, 1, 64, 64)[:n],
                    f_recon_batch.view(batch_size, 1, 64, 64)[:n],
                    (f_data.view(batch_size, 1, 64, 64)[:n] -
                     f_recon_batch.view(batch_size, 1, 64, 64)[:n]),
                    sig_plot.view(batch_size, 1, 64, 64)[:n] * 5,
                    err_rec.view(batch_size, 1, 64, 64)[:n],
                    seg.view(batch_size, 1, 64, 64)[:n]
                ])
                save_image(comparison.cpu(),
                           'results/roc' + str(i) + '.png',
                           nrow=n)

            if i == 0:
                rec_error_all = rec_error
            else:
                rec_error_all = torch.cat([rec_error_all, rec_error])
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return rec_error_all


if __name__ == "__main__":
    rec_error_all = Validation(X)
    y_true = X[0:15 * 200, ::2, ::2, 3]
    y_true = np.reshape(y_true, (-1, 1))

    maskX = np.reshape(X[0:15 * 200, ::2, ::2, 2], (-1, 1))
    y_true = y_true*(maskX > 0)

    y_probas = (rec_error_all).to('cpu')
    y_probas = y_probas.numpy()
    y_probas = np.reshape(y_probas, (-1, 1))
    y_true = y_true.astype(int)

    print(np.min(y_probas))
    print(np.max(y_probas))

    y_probas = np.reshape(y_probas, (-1, 1, 64, 64))
    y_probas = np.reshape(y_probas, (-1, 1))
    y_probas = y_probas*(maskX > 0)

    fpr, tpr, th = metrics.roc_curve(y_true, y_probas)
    L = fpr / tpr
    best_th = th[fpr < 0.05]
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

    y_probas[y_probas >= best_th[0]] = 1
    y_probas[y_probas < best_th[0]] = 0

    y_probas = np.reshape(y_probas, (-1, 64 * 64 * 20))
    y_true = np.reshape(y_true, (-1, 64 * 64 * 20))

    dice = 0
    for i in range(y_probas.shape[0]):
        seg = y_probas[i, :]
        gth = y_true[i, :]
        dice += np.sum(seg[gth == 1]) * 2.0 / (np.sum(gth) + np.sum(seg))
    print((dice) / y_probas.shape[0])
