import numpy as np
from spydernet import train_model
from datautils import read_data


def main():
    """ Main script that calls the functions objects"""
    data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'

    train_data = read_data(
        study_dir=data_dir, nsub=5, psize=[16, 16], npatch_perslice=64)

    model = train_model(train_data)


if __name__ == "__main__":
    main()
