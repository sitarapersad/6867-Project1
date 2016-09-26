import pdb
import random
import pylab as pl
import numpy as np
import ski

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

global X
global Y

X, Y = lassoTrainData()

print X, X.shape
print '...'
print Y
def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')


#def computePhiX(phi, X):
#    '''
#    Given an n x d dataset and a basis function, compute phi(X)
#    
#    @param: phi - list functions to form new basis
#    @param: X - n x d dataset
#    
#    @return phi(X) - m x n array, where columns are phi(x_i)
#    '''
#    n, d = X.shape
#    phi_X = []
#    for i in range(n):
#        x = X[i]
#        phi_x_i = []
#        for f in phi:
#            phi_x_i.append(f(x))
#        phi_X.append(phi_x_i)
#    phi_X = np.array(phi_X).T 
#    return phi_X
#    
#def lassoRegression(w,L):
#    '''
#    @param: L - lambda value to fine tune regularization
#    
#    @return:
#    '''
#    n,d = X.shape #n = number of data pts; d = dimension of data
#    phi = [lambda x: x]+[lambda x: np.sin(0.4*x*i) for i in range(1,14)]    
#    
#    #compute phi(X)
#    phi_X = computePhiX(phi, X)
#    return 1./2 * np.linalg.norm(Y-w.T*phi_X) + L*np.linalg.norm(w, ord=1)
#    
