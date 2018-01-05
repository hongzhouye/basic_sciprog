"""
This script provides functions for linear least square fitting
"""


import numpy as np
import sys
from matplotlib import pyplot


def lls_fit(X, Y):
    """Given X and Y, solve A that optimizes the 2-norm difference between
    A.T @ X and Y.

    Attributes:
        X (np.ndarray):
            Dimension: n-by-N. X = [x^1, x^2, \cdots{}, x^N] where x^i is the
            i-th data point vector.
        Y (np.ndarray):
            Dimension: m-by-N. Y = [y^1, y^2, \cdots{}, y^N] where y^i is the
            i-th target vector.

    Returns:
        A (np.ndarray):
            Dimension: n-by-m.
    """
    if X.ndim == 1:
        A = X@Y / (X@X)
    else:
        A = np.linalg.pinv(X@X.T) @ X@Y.T
    return A


def lls_predict(A, X):
    """Given X and A, compute predicted value Y = A.T@X
    """
    if X.ndim == 1:
        return A * X
    else:
        return A.T @ X


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: dataX dataY")
        sys.exit(1)
    
    # read data
    X = np.loadtxt(sys.argv[1])
    Y = np.loadtxt(sys.argv[2])

    # sort data
    if X.ndim == 1:
        inds = np.argsort(X)
        Xnew, Ynew = [], []
        for i in inds:
            Xnew.append(X[i])
            Ynew.append(Y[i])
        X = np.array(Xnew)
        Y = np.array(Ynew)
    
    # put data into the desired format
    if X.ndim != 1:
        X = X.T
        Y = Y.T
    
    # fit
    A = lls_fit(X, Y)
    
    # predict
    x = np.arange(0, 20, 0.5)
    pred_Y = lls_predict(A, x)
    
    # plot and compare
    if X.ndim == 1:
        pyplot.plot(X, Y, 'ko', label="data")
        pyplot.plot(x, pred_Y, 'r', label='fitted')
        pyplot.show()
