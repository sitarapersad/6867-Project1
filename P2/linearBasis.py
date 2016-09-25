import matplotlib.pyplot as plt
import pylab as pl
import numpy as np 

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)

#helper function phi for computerPolynomialWeight(X, Y, M)
def phi_polynomial(x, i): 
    '''
    @param: x = function argument
    @param: i = exponent
    '''
    return x**i

def computePolynomialWeight(X, Y, M):
    '''
    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: M - maximum order of polynomial basis

    @return: ? x ? numpy array which is the MLE for the weight vector
    '''
    #Variables here
    N = len(X) 
    #Generate floating-point zero matrix of size NxM to populate with phi_polynomial function
    design_matrix = numpy.zeros((N, M+1))

    #Populate Matrix
    for i in range(len(x)):
        for j in range(M):
            design_matrix[i, j] = phi_polynomial(X[i], j)
    
    #Calculate Max-Likelihood Weight Vector


    #generate the phi vector based on the maximum order where phi_i = x^i
    #phi = np.array[lambda x: x**i for i in range(M+1)]

    #figure out how to do this the numpy way; see vectorize??
    #design_matrix = np.array([[phi[i](j) for i in range(M+1)] for j in x]).T

    #moore-penrose inverse = (phi.T x phi)^-1 x phi.T 
    moore_penrose_inverse = np.dot(np.linalg.inv(np.dot(design_matrix.T,design_matrix)), design_matrix.T)
    assert moore_penrose_inverse.shape = (M, 1)

    return np.dot(moore_penrose_inverse, Y)


def computeSSE(X, Y, M_list, w):
    '''
    Computes the sum of squares error (SSE) and derivative for a data
    set and a hypothesis, specified by the list of M polynomial basis 
    functions and a weight vector

    Least squares error is given by J(x,y,w) = 1/2 sum_over_data (Y-w.T*phi(X_n))**2

    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: M_list - ordered list of polynomial basis functions phi_M(x) = x^m
    @param: w - ? x ? array weight vector

    @return: squared_error 
    '''
    TODO: AUGMENT X WITH OFFSET??

    n,d = X.shape
    assert d == 1 #Data is real valued

    #create phi function to compute phi's guess of Y
    def phi(x):
        return sum(f(x) for f in M_list)

    sqrd_err = 0
    for i in range(1, n+1):
        sqrd_err += np.power(Y[i] - np.dot(w.T,phi(X[i])), 2.)
    
    return 0.5*sqrd_err

def computeSSE_prime(X, Y, M_list, w):
    '''
    Computes the sum of squares error (SSE) and derivative for a data
    set and a hypothesis, specified by the list of M polynomial basis 
    functions and a weight vector

    Least squares error grad is given by J(x,y,w) = sum_over_data (y[i]-w.T*phi(x[i]))*phi(x[i])
    see Eq'n 3.23 Bishop
    
    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: M_list - ordered list of polynomial basis functions phi_M(x) = x^m
    @param: w - ? x ? array weight vector

    @return: derivative of squared error 
    '''
    n,d = X.shape
    assert d == 1 #Data is real valued

    #create phi function to compute phi's guess of Y
    def phi(data):
        return sum(f(data) for f in M_list)

    sqrd_err = 0
    for i in range(n):
        sqrd_err += phi(x[i])* (y[i] - np.dot(w.T,phi(x[i])) 
    return sqrd_err



# 4: Using cosine as our phi function

def phi_cos(x, i):
    return np.cos(i*np.pi*x)

#M = 8
def computeCosWeight(X, Y, M = 8):
    '''
    @param: X - n x 1 array of n 1-dimensional data points
    @param: Y - n x 1 array of corresponding y vals
    @param: M - maximum order of polynomial basis (set default to 8)

    @return: ? x ? numpy array which is the MLE for the weight vector
    '''
    #Variables here
    N = len(X) 
    #Generate floating-point zero matrix of size NxM to populate with phi_polynomial function
    design_matrix = numpy.zeros((N, M+1))

    #Populate Matrix
    for i in range(len(x)):
        for j in range(M):
            design_matrix[i, j] = phi_cos(X[i], j)
    
    #Calculate Max-Likelihood Weight Vector
    #moore-penrose inverse = (phi.T x phi)^-1 x phi.T 
    moore_penrose_inverse = np.dot(np.linalg.inv(np.dot(design_matrix.T,design_matrix)), design_matrix.T)
    assert moore_penrose_inverse.shape = (M, 1)

    return np.dot(moore_penrose_inverse, Y)

