##conv AE
import numpy as np
import tensorflow as tf
#from BasicAutoencoder import DeepAE as DAE
from shrink import l21shrink 
from VAE import Vauto_encoder
class RobustL21Autoencoder(object):
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016
    Updated to python3
    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    Improve:
        1. fix the 0-cost bugs
    """
    def __init__(self, input_shape, original_dim,loss,lambda_=1.0, error = 1.0e-8):
        """
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.error = error
        self.errors=[]
        (self.AE ,self.hyden) = Vauto_encoder(input_shape,original_dim,loss)


    def fit_T(self, X, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=133):
        lamb=self.lambda_
        ## initialize L, S
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        ##LS0 = self.L + self.S
        ## To estimate the size of input X

        for it in range(iteration):
            print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder ## get optmized L
            self.AE.fit(self.L,
                        epochs = inner_iteration,
                        batch_size = batch_size,
                        shuffle=True)
            self.AE.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/L12/models/AE_my_model_%f.h5' % lamb)
            self.L = self.getRecon(X = self.L) 
            ## alternating project, now project to S and shrink S
            self.S = l21shrink(self.lambda_, (X-self.L).T).T
        return self.L , self.S
    
    def transform(self, X):
        L = X - self.S
        return self.hyden.predict(L)
    
    def getRecon(self, X):
        return self.AE.predict(X)
    
if __name__ == "__main__":
    x = np.load(r"data.npk")[:500]
    image_size = x.shape[1]
    original_dim = image_size 
    input_shape = (original_dim, )
    #AE ,hyden = Vauto_encoder(input_shape,original_dim,loss)
    rae = RobustL21Autoencoder(input_shape=input_shape, original_dim=original_dim,loss='MSE',lambda_= 20)
    #AE.fit(x,x,
                        #epochs = inner_iteration,
                        #batch_size = batch_size,
                        #shuffle=True)
    L, S = rae.fit_T(x, inner_iteration = 60, iteration = 5)