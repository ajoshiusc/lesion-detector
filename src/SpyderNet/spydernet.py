# Author: Anand A Joshi (ajoshi@usc.edu)


def spyder_net():
    
    print('hello')


def train_model(model1, data):

    return model1


def test_model(model, data):

    return predictions

def get_neural_net(self, isize=[32, 32], subc_size=31870):
    """VGG model with one FC layer added at the end for continuous output"""
    lh_input, lh_out = get_mynet(isize, 'lh_')
    rh_input, rh_out = get_mynet(isize, 'rh_')

    subco_input = Input(shape=(subc_size, 36), dtype='float32')
    fc = Flatten()(subco_input)
    subco_out = Dense(256, activation='relu')(fc)

    cc = concatenate([lh_out, rh_out, subco_out], axis=-1)
    cc = Dense(64, activation='relu')(cc)
    out_theta = Dense(3)(cc)

    print("==Defining Model  ==")
    model = Model(
        inputs=[lh_input, rh_input, subco_input], outputs=[out_theta])
    optz = adam(lr=1e-4)  #, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=optz, loss=losses.mean_squared_error, metrics=['mse'])

    return model
