import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    return (1. / (1. + np.exp(-x)))


def beta_divergence(y_true, y_pred, sample_weights=None, beta=0.5):#.0000001):
    f = sigmoid(np.array(y_true))
    g = sigmoid(np.array(y_pred))
    g = np.reshape(g, f.shape)

    C = f**(1. + beta) - (
        (beta + 1.) / beta) * g * f**beta + (1. / beta) * g**(1. + beta) + (
            1. - f)**(1. + beta) - ((beta + 1.) / beta) * (1. - g) * (
                1. - f)**beta + (1. / beta) * (1. - g)**(1. + beta)

    return np.sum(C)


def mean_absolute_percentage_error(y_true, y_pred, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    if np.any(y_true == 0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true == 0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)

    if type(sample_weights) == type(None):
        return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return (100 / sum(sample_weights) * np.dot(sample_weights, (np.abs(
            (y_true - y_pred) / y_true))))


def objective_function(beta, X, Y):
    error = loss_function(np.matmul(X, beta), Y)
    return (error)


class CustomLinearModel:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """

    def __init__(
            self,
            loss_function=beta_divergence,  #mean_absolute_percentage_error,
            X=None,
            Y=None,
            sample_weights=None,
            beta_init=None,
            regularization=0.00012):
        self.regularization = regularization
        self.beta = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.beta_init = beta_init

        self.X = X
        self.Y = Y

    def predict(self, X):
        prediction = np.matmul(X, self.beta)
        return (prediction)

    def model_error(self):
        error = self.loss_function(
            self.predict(self.X), self.Y, sample_weights=self.sample_weights)
        return (error)

    def l2_regularized_loss(self, beta):
        self.beta = beta
        return(self.model_error() + \
               sum(self.regularization*np.array(self.beta)**2))

    def fit(self, X, Y, maxiter=250):
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        self.X = X
        self.Y = Y
        if type(self.beta_init) == type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.array([1] * self.X.shape[1])
        else:
            # Use provided initial values
            pass

        if self.beta != None and all(self.beta_init == self.beta):
            print(
                "Model already fit once; continuing fit with more itrations.")

        res = minimize(
            self.l2_regularized_loss,
            self.beta_init,
            method='BFGS',
            options={'maxiter': 500})
        self.beta = res.x
        self.beta_init = self.beta


#loss_function = mean_absolute_percentage_error