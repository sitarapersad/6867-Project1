import pdb
import random
import pylab as pl

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

#FROM PROBLEM 2: TO CALCULATE w 
def phi_polynomial(x, i): 
    '''
    @param: x = function argument
    @param: i = exponent
    '''
    return x**i


def ridge_regression(X, Y, L, M): 
    '''
    Implementing ridge regression as per Bishop Eq. 3.27 and Eq. 3.28.

    BISHOP 3.27: Full Error = 0.5*[(sum from n=1 to N of: [Y[n] - w.Transpose*phi(x[n])]^2 + L/2(w.Transpose * w)
    BISHOP 3.28: w = inverse L*I + phi.T * phi) * phi.T * Y 

    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: L - regularization coefficient
    @param: M - maximum order of polynomial basis
    
    @return: Full Error
    '''

    #Step 1: calculate w. (using Bishop 3.28)
  
    #Variables here
    N,d = X.shape
    assert d == 1 #Data is real valued 
    
    I = np.identity(M+1)
     
    #Generate floating-point zero matrix of size N x (M+1) to populate with phi_polynomial function
    phi_matrix = numpy.zeros((N, M+1))

    #Populate Matrix
    for i in range(N):
        for j in range(M+1):
            phi_matrix[i, j] = phi_polynomial(X[i], j)
    
    #calculate w: np.dot(np.linalg.inv(np.dot(design_matrix.T,design_matrix)), design_matrix.T)
    w = np.dot(np.linalg.inv(np.dot(L, I) + np.dot(phi_matrix.T, phi_matrix)), np.dot(phi_matrix.T, Y))
    
    E_d = 0
    #STEP 2: Calculate E_d (sum of squares from 1 -> N)
    for i in range(N):
        E_d += 0.5 * np.power(Y[i] - np.dot(w.T, [np.power(X[i],j) for j in range(M+1)]), 2.)
    
    #STEP 3: Calculate L*E_w
    E_w = 0.5 * L * np.dot(w.T, w)

    #STEP 4: Return E_d + L*E_w

    return E_d + E_w