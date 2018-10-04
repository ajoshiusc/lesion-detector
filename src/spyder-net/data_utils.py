import glob
import nilearn as nl
import os


def read_data(study_dir, nsub, npatch_perslice):
    dirlist = glob.glob(study_dir)
    for subj in dirlist:
        t1file = os.path.join(subj,'T1.nii.gz')
        t2file = os.path.join(subj,'T2.nii.gz')
        fl = os.path.join(subj,'FLAIR.nii.gz')

        t1 = nl.load(t1file)
        t2 = nl.load(t2file)
        flair = nl.load(flfile)

        # Read coronal slices

        create image

        create random patches

        return patch_data # npatch x width x height x channels
