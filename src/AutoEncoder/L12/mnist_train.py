from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)