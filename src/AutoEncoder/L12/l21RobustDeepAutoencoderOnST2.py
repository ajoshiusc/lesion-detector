##conv AE
import numpy as np
import tensorflow as tf
#from BasicAutoencoder import DeepAE as DAE
from shrink import l21shrink 
from test_AE import auto_encoder
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
    def __init__(self, input_size, lambda_=1.0, error = 1.0e-8):
        """
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.error = error
        self.errors=[]
        (self.AE ,self.hyden) = auto_encoder(input_size)


    def fit_T(self, X, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=133):
        
        ## initialize L, S
        lamb=self.lambda_
        self.lambda_=self.lambda_*X.shape[0]
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        ##LS0 = self.L + self.S
        ## To estimate the size of input X

        for it in range(iteration):
            print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder ## get optmized L
            self.AE.fit(self.L,self.L,
                        epochs = inner_iteration,
                        batch_size = batch_size,
                        shuffle=True)
            self.L = self.getRecon(X = self.L)
            self.AE.save('/big_disk/akrami/git_repos/lesion-detector/src/AutoEncoder/L12/models/AE_my_model_%f.h5' % lamb)
            ## alternating project, now project to S and shrink S
            temp=(X[:,:,:,0] - self.L[:,:,:,0])
            temp=temp.reshape(temp.shape[0],784)
            tempS = l21shrink(self.lambda_, temp.T).T
            self.S=tempS.reshape(tempS.shape[0],28,28,1)


        return self.L[:,:,:,0] , self.S[:,:,:,0]
    
    def transform(self, X):
        L = X - self.S
        return self.hyden.predict(L)
    
    def getRecon(self, X):
        return self.AE.predict(X)
    
if __name__ == "__main__":
    xin = np.load(r"data.npk")[:500]
    x=xin.reshape(xin.shape[0],28,28,1)
    rae = RobustL21Autoencoder(lambda_= 20, input_size=28)
    L, S = rae.fit_T(x, inner_iteration = 60, iteration = 5)