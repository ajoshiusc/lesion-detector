
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from lin_regress import CustomLinearModel

import pandas as pd

X = np.linspace(0.0001,0.999,100)
Y = 0.3 * X #+ .7

#X = X + np.random.randn(X.shape[0])
Y = Y + .01 * np.random.randn(X.shape[0])
Y[90:95] = Y[90:95] - .5
#X[5:10] = X[5:10] - 200

X_orig1 = X.copy()
Yorig = Y.copy()
plt.scatter(X_orig1, Yorig, color='black')

for lamd in np.arange(0.1, 60, 5):
    #diabetes = datasets.load_diabetes()
    X = X_orig1.copy()
    Y = Yorig.copy()

    indices = (0)

    X = X[:, None]  #diabetes.data[:-20, indices][:,None]
    #X_test = X[-20:][:, None]  #diabetes.data[-20:, indices][:,None]
    y_train = Y[:, None]  #diabetes.target[:-20]
    #y_test = Y[-20:][:, None]  #diabetes.target[-20:]

    ols = CustomLinearModel()
    ols.fit(X, y_train)

#    ols_inv = linear_model.LinearRegression()
#    ols_inv.fit(y_train, X)

    print(ols.beta)

    

#    S = 0 * Y

#    Y = Ld

    y_pred = ols.predict(X)

    plt.plot(X, y_pred, linewidth=1, label = str(lamd))


plt.xticks(())
plt.yticks(())
plt.legend(loc = 'best')

plt.show()
