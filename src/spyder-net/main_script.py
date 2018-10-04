# ||AUM||
# ||Shree Ganeshaya Namaha||



# ||AUM||
# ||Shree Ganeshaya Namaha||
import numpy as np
from spyder_net import train_model
from data_utils import read_data

def main():
    """ Main script that calls the functions objects"""
    data_dir = '/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'


    model = train_model(data_dir=train_data_dir, csv_file=csv_file)
    y, ypred = cp.predict(data_dir=test_data_dir, csv_file=csv_file)



if __name__ == "__main__":
    main()
