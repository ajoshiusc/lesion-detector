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
from l12models import simple_linear_model
simple_linear_model

def proxl1(S, lamd):
    print(S.shape)
    S[S > lamd] -= lamd
    S[S < -lamd] += lamd
    S[(-lamd < S) & (S < lamd)] = 0
    print(S.shape)
    return S


X = np.arange(100)
Y = 0.3 * X + .7
print(Y.shape)
#X = X + np.random.randn(X.shape[0])
Y = Y + np.random.randn(X.shape[0])
Y[5:10] = Y[5:10] + 120
#X[5:10] = X[5:10] - 200

X_orig1 = X.copy()
Yorig = Y.copy()
plt.scatter(X_orig1, Yorig, color='black')
model=simple_linear_model(1)
for lamd in np.arange(0, 0.095, 0.005):
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
    print(X.shape)
    for j in range(500):
        Ld = Y - S
        print(Ld.shape)

        model.fit(X,Ld,

            epochs=20,

            batch_size=8,

            shuffle=True,)

        Ld=model.predict(X)    
        print(Ld.shape) 
        print(Y.shape)  
        print(S.shape) 
        S = Y - Ld[:,0]
        print(S.shape) 
        S = proxl1(S, lamd)

    Y = Ld

    y_pred = model.predict(X)
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