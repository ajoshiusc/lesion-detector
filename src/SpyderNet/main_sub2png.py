from read_test_data import subdata2png


def main():
    """ Main script that calls the functions objects"""
    data_dir = '/home/ajoshi/Desktop/lesion_subjects'
    subid = 'TBI_INVKZ324MM1'
    #subid = 'TBI_INVJH729XF3'


    subdata2png(study_dir=data_dir, subid=subid, axis=0)

    # Read data


if __name__ == "__main__":
    main()
