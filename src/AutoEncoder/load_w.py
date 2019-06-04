from keras.models import load_model
from deep_auto_encoder2 import FLAIR_loss
alpha=1
model=load_model('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_bactchnorm.h5',custom_objects={'MSE_FLAIR': FLAIR_loss(alpha)})
model.save_weights('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/models/tp_model_200_512_merryland_30_MSEF2_bactchnorm_wights.h5')