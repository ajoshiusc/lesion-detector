import numpy as np
from tqdm import tqdm
def slice2vol_pred(model, vol_data, im_size, step_size):
   

# model_pred: predictor that gives 2d images as outputs
# vol_data this is 3d images + 4th dim for different modalities
# im_size number with size of images (for 64x64 images) it is 64
# step_size : size of offset in stacked reconstruction

    out_vol = np.zeros(vol_data.shape[:]) # output volume
    indf = np.zeros(vol_data.shape[:]) # dividing factor
    vol_size = vol_data.shape

    print('Putting together slices to form volume')
    for j in tqdm(range(0, vol_size[1] - im_size, step_size)):
        for k in range(0, vol_size[2] - im_size, step_size):
            out_vol[:, j:im_size + j, k:im_size +
                    k,:] += model.predict([
                        vol_data[:, j:im_size + j, k:im_size + k,:],
                    ]).squeeze()
            indf[:, j:im_size + j, k:im_size + k,:] += 1


#        print(j ,'out of', vol_size[1] - im_size)

    out_vol = out_vol / (indf + 1e-12)  #[...,None]

    return out_vol