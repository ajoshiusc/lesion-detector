from data_utils import read_data
data_dir='/big_disk/ajoshi/fitbir/preproc/tracktbi_pilot'
data = read_data(
study_dir=data_dir, nsub=1, psize=[35, 35], npatch_perslice=8)


