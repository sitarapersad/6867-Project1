import pylab as pl
import numpy
import pandas 

def getData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y) 


def gradientDescent(x, y, objective_fn, gradient_fn, initial_guess, step_size, convergence):

    '''
    Implements batch gradient descent 

    Batch update function: w(t+1) = w(t) - n grad_E(w(t)) 

    !! QUESTIONS: Do we input an error function and compute the gradient from that
    !! How do we compute it? aka no gradient_fn parameter needed?

    @param: x - n x d numpy array of n d-dimensional data points
    @param: y - n x 1 numpy array of n labels/points 
    @param: objective_fn - 
    @param: gradient_fn - 
    @param: initial_guess (w(0)) - 
    @param: step_size (n) - 
    @param: convergence - threshold 

    @return: best_guess
    '''

    ## SKETCH OF ALGORITHM; WILL NOT ACTUALLY WORK LEL
    w = initial_guess
    converged = False
    while not converged:

        # perform batch gradient descent 
        w_new -= step_size*gradient_fn(w)
        if abs(w_new - w) < convergence:
            converged = True

    return best_guess

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


    return gradient 


def stochasticGradientDescent(x, y, ):
    '''
    Iplements stochastic gradient 

    Stochastic update function: w(t+1) = w(t) - n grad_E(w(t)) [for some random data point]

    !! QUESTIONS: Do we input an error function and compute the gradient from that
    !! How do we compute it? aka no gradient_fn parameter needed?

    @param: x - n x d numpy array of n d-dimensional data points
    @param: y - n x 1 numpy array of n labels/points 
    @param: objective_fn - 
    @param: gradient_fn - 
    @param: initial_guess (w(0)) - 
    @param: step_size (n) - 
    @param: convergence - threshold 

    @return: best_guess
    '''