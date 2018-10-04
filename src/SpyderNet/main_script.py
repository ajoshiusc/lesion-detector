import numpy as np
from spydernet import train_model
from datautils import read_data


def main():
    """ Main script that calls the functions objects"""
    data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'

    train_data = read_data(data_dir, nsub=5, npatches_perslice=16)

    model = train_model(data_dir=train_data_dir, csv_file=csv_file)


if __name__ == "__main__":
    main()


