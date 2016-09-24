import pylab as pl
import numpy as np
import pandas 

def getData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y) 

# define functions to test gradient descent on

def computeGaussian(x, mu, sigma):
    '''
    
    '''
    return 1.0/(2*pi) * np.exp(-np.power(x-mu, 2.)/(2*np.power(sigma,2.)))

def differentiateGaussian(x, mu, sigma):

    return None 

def computeQuadBowl(x):

    return None 

def differentiateQuadBowl(x):

    return None 


# Implementation 
def gradientDescent(objective_fn, gradient_fn, initial_guess, step_size, convergence):

    '''
    Implements batch gradient descent 

    Batch update function: w(t+1) = w(t) - n grad_E(w(t)) 

    @param: objective_fn - loss function 
    @param: gradient_fn - gradient of loss function; if not specified, compute numerical approx
    @param: initial_guess (w(0)) - 
    @param: step_size (n) - 
    @param: convergence - if successive function values differ below this threshold, stop iterating 

    @return: best_guess
    '''
    w = initial_guess
    converged = False
    num_iters = 0

    while not converged: # perform batch gradient descent until convergence
        num_iters += 1
        if gradient_fn != None :
            w_new -= step_size*np.array(gradient_fn(w))
        else:
            # if no gradient function specified, estimate gradient by finite differences method
            ## TODO!
            pass
        if abs(objective_fn(w_new) - objective_fn(w)) < convergence: 
            converged = True
        w = w_new

    best_guess = w
    best_value = objective_fn(w)
    return best_guess, best_value

def approximateGradient(point, approx_fn, delta):
    '''
    Calculates the approximate gradient at a point using the finite
    differences method.

    Formula: f'(x) ~= 1/d * f(x+d/2) - f(x-d/2)

    @param: point - n-dimensional point at which we approximate the grad
    @param: approx_fn - function whose gradient we approximate
    @param: delta - step size for finite differences method

    @return: gradient - float which is an approximation of the gradient at point
    '''

    ## TODO: WHAT IS DELTA FOR A VECTOR X?
    gradient = 1.0/delta * ( approx_fn(point+0.5*delta) - approx_fn(x-0.5*delta) ) 
    return gradient 


def stochasticGradientDescent(x, y, ):
    '''
    Iplements stochastic gradient 

    Stochastic update function: w(t+1) = w(t) - n grad_E(w(t)) [for some random data point]

    @param: x - n x d numpy array of n d-dimensional data points
    @param: y - n x 1 numpy array of n labels/points 
    @param: objective_fn - 
    @param: gradient_fn - 
    @param: initial_guess (w(0)) - 
    @param: step_size (n) - 
    @param: convergence - threshold 

    @return: best_guess

    This is the exact same procedure as batch gradient descent, except the gradient_fn is 
    computed over one data point.

    COPY PASTA TIME BOYZ
    '''