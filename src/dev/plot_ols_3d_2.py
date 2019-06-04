#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=========================================================
Sparsity Example: Fitting only features 1  and 2
=========================================================

Features 1 and 2 of the diabetes-dataset are fitted and
plotted below. It illustrates that although feature 2
has a strong coefficient on the full model, it does not
give us much regarding `y` when compared to just feature 1

"""
print(__doc__)

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model


def proxl1(S, lamd):
    S[S > lamd] -= lamd
    S[S < -lamd] += lamd
    S[(-lamd < S) & (S < lamd)] = 0
    return S


X = np.arange(100)
Y = 0.3 * X + .7

#X = X + np.random.randn(X.shape[0])
Y = Y + np.random.randn(X.shape[0])
Y[5:10] = Y[5:10] + 120
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

    ols = linear_model.LinearRegression()
    ols.fit(X, y_train)

#    ols_inv = linear_model.LinearRegression()
#    ols_inv.fit(y_train, X)

    print(ols.intercept_, ols.coef_)

    

    S = 0 * Y

    for j in range(500):
        Ld = Y - S

        ols.fit(X, Ld)
#        ols_inv.fit(y_train, Ld)

        # Make predictions using the testing set
        Ld = ols.predict(X)
 #       Ld = ols_inv.predict(y_pred)
        S = Y - Ld
        S = proxl1(S, lamd)

    Y = Ld

    y_pred = ols.predict(X)
    # The coefficients
    print('Coefficients: \n', ols.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_train, y_pred))

    # Plot outputs
#    plt.scatter(X, Y, color='green')

    plt.plot(X, y_pred, linewidth=1, label = str(lamd))


plt.xticks(())
plt.yticks(())
plt.legend(loc = 'best')

plt.show()
