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

X = np.arange(100)
Y = 0.3 * X+.7

X = X + np.random.randn(X.shape[0])
Y = Y + np.random.randn(X.shape[0])
#Y[80:90] = Y[80:90] + 20
#X[80:90] = X[80:90] - 20

#diabetes = datasets.load_diabetes()
indices = (0)

X_train = X[:, None]  #diabetes.data[:-20, indices][:,None]
#X_test = X[-20:][:, None]  #diabetes.data[-20:, indices][:,None]
y_train = Y[:, None]  #diabetes.target[:-20]
#y_test = Y[-20:][:, None]  #diabetes.target[-20:]

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

ols_inv = linear_model.LinearRegression()
ols_inv.fit(y_train, X_train)

print(ols.intercept_, ols.coef_)

# Make predictions using the testing set
y_pred = ols.predict(X_train)
X_pred = ols_inv.predict(y_pred)



# The coefficients
print('Coefficients: \n', ols.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_train, y_pred))

# Plot outputs
plt.scatter(X_train, y_train, color='black')
plt.scatter(X_pred, y_pred, color='green')

plt.plot(X_train, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
