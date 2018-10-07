import numpy as np
from spydernet import train_model, mod_indep_rep
from datautils import read_data
import matplotlib.pyplot as plt


def main():
    """ Main script that calls the functions objects"""
    data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'

    data = read_data(
        study_dir=data_dir, nsub=5, psize=[32, 32], npatch_perslice=256)

    train_data = data #[0:-5, :, :, :]
    model = train_model(train_data)

    test_data = data[90:95, :, :, :]
    I, pred = mod_indep_rep(model, test_data)

    plt.figure()
    for j in range(5):
        plt.subplot(2, 5, j + 1)
        plt.imshow(I[j, :, :, :].squeeze())
        plt.subplot(2, 5, 5 + j + 1)
        plt.imshow(test_data[j, :, :, 0].squeeze())
        plt.subplot(2, 5, 10 + j + 1)
        plt.imshow(pred[j, :, :, 0].squeeze())

    plt.show()

    print(test_data)
    print(I)

#    for j in range(5):
#        plt.matshow(I[j,:,:,:].squeeze())

#    plt.show()

if __name__ == "__main__":
    main()
