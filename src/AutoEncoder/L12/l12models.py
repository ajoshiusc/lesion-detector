import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

def simple_linear_model(inputshape):
    inputs=Input(shape=(inputshape,))
    x=Dense(1,activation='linear')(inputs)
    prediction=Dense(1,activation='linear')(inputs)
    model=Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
