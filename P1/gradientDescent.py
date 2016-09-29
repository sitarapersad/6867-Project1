import pylab as pl
import numpy as np
#import gradientDescent as gd
import loadParametersP1 as params

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
    Evaluates the Gaussian at a given point, given by:
    f(x) = = 1/sqrt(2pi)^n|sigma|exp[ -1/2(x-u).T sigma^-1 (x-mu) ]
    
    @param: x
    @param: mu
    @param: sigma
    
    @return: result- scalar which is the result of evaluating the function
    '''
    if isinstance(x, float):
        d = 1
    else:
        x = np.array(x)
        d = float(x.shape[0])
    mu = np.array(mu)
    sigma = np.array(sigma)
       
    exponent = np.dot(np.linalg.inv(sigma),(x-mu))
    exponent = np.dot((x-mu).T,exponent)
    return -1./(np.linalg.det(sigma)**0.5*(np.power(2*np.pi, d/2))) * np.exp(-1./2 * exponent )

def differentiateGaussian(x, mu, sigma):
    '''
    Computes the derivative of the Gaussian, given by:
    df/dx = f(x)(sigma^-1)(x-mu)
    
    @param: x
    @param: mu
    @param: sigma
    
    @return
    '''
    # print 'gaussian deriv: ',computeGaussian(x, mu, sigma)*np.dot(np.linalg.inv(sigma),(x-mu)), 'end'
    return -1.*computeGaussian(x, mu, sigma)*np.dot(np.linalg.inv(sigma),(x-mu))

def computeQuadBowl(x,A,b):
    '''
    Evaluates the derivative of thequadratic bowl at a point, given by:
    f(x) = 1/2 x^T(Ax-2b)

    @param: x - point vector
    @param: A - 
    @param: b - 
    
    @output: result - a vector with the dimensions of x which is the result of the function evaluation
    '''
    return np.dot(x,(1/2*np.dot(A,x)-b))
    
def differentiateQuadBowl(x,A,b):
    '''
    Evaluates the derivative of thequadratic bowl at a point, given by:
    f(x) = Ax-b
    
    @param: x -
    @param: A - 
    @param: b - 
    
    @output: result - a vector with the dimensions of x which is the result of the function evaluation
    '''
    return np.dot(A,x) - b

def computeSquaredLoss(X, Y, theta):
    '''
    Computes the squared loss of a parameter, theta, on a dataset (X,Y) using the formula
    f(theta) = ||X.theta-Y||^2
    
    @param: X -
    @param: Y - 
    @param: theta - 
    
    @output: sqrd_loss - scalar which is the squared error 
    '''
    return np.linalg.norm(np.dot(X,theta)-Y)
    
def differentiateSquaredLoss(X,Y,theta):
        '''
    Computes the squared loss of a parameter, theta, on a dataset (X,Y) using the formula
    f(theta) = ||X.theta-Y||^2
    
    @param: X -
    @param: Y - 
    @param: theta - 
    
    @output: gradient - vector which is the squared error derivative wrt theta 
    '''
        gradient = np.zeros(theta.shape)
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            gradient += x* (theta.T*x - y)
        return gradient


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
    @return: best_value
    @return: guess_evolution
    @return: fxn_evolution
    @return: norm_evolution
    '''
    w = np.array(initial_guess,dtype=np.float64).T
    
    converged = False
    num_iters = 0
        
    guess_evolution = []
    fxn_evolution = []
    norm_evolution = []
    
    while not converged: # perform batch gradient descent until convergence
        num_iters += 1
        w_new = w 
        if gradient_fn != None :
            w_new = w - step_size*np.array(gradient_fn(w))
            norm_evolution.append(np.linalg.norm(gradient_fn(w)))
        else:
            # if no gradient function specified, estimate gradient by finite differences method
            w_new = w - step_size*np.array(approximateGradient(w, objective_fn, 0.00001))
            norm_evolution.append(np.linalg.norm(approximateGradient(w,objective_fn, 0.00001)))
        if abs(objective_fn(w_new) - objective_fn(w)) < convergence: 
            converged = True
        
        # keep track of how the guess and function change as we run gradient descent  
        guess_evolution.append(w)
        fxn_evolution.append(objective_fn(w))

        # reset w to the newest version
        w = w_new
    
    guess_evolution.append(w)
    fxn_evolution.append(objective_fn(w))
    norm_evolution.append(np.linalg.norm(gradient_fn(w)))
    
    best_guess = w
    best_value = objective_fn(w)
    
    return best_guess, best_value, guess_evolution, fxn_evolution, norm_evolution

def approximateGradient(point, approx_fn, delta):
    '''
    Calculates the approximate gradient at a point using the finite
    differences method.

    Formula: f'(x) ~= 1/d * f(x+d/2) - f(x-d/2)

    @param: point - 1-dimensional point at which we approximate the grad
    @param: approx_fn - function whose gradient we approximate
    @param: delta - step size for finite differences method

    @return: gradient - float which is an approximation of the gradient at point
    '''
#    gradient = np.zeros(point.shape)
#    print gradient
#    for i in xrange(point.shape[0]):
#        gradient[i] = approx_fn(point[i]+0.5*delta) - approx_fn(point[i]-0.5*delta) 
    # convert delta scalar to delta vector
    delta_val = delta
    if isinstance(point, float):
        delta = np.zeros(1)
        gradient = np.zeros(1)
    else:
        delta = np.zeros(point.shape)
        gradient = np.zeros(point.shape)
    delta.fill(delta_val)
    
    
    gradient.fill(approx_fn(point+0.5*delta) - approx_fn(point-0.5*delta) ) 

    return 1./delta_val*gradient 


def stochasticGradientDescent(x, y, objective_fn, gradient_fn, initial_guess, step_size, convergence):
    '''
    Iplements stochastic gradient descent

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
    computed over a randomly chosen data point.
    '''
    w = np.array(initial_guess,dtype=np.float64).T

    converged = False
    num_iters = 0
        
    n,d = X.shape
    
    guess_evolution = []
    fxn_evolution = []
    norm_evolution = []

    while not converged: # perform batch gradient descent until convergence
        num_iters += 1
        w_new = w 

        # randomly choose data point to perform update with
        random_index = np.random.randint(0,d)
        
        # compute gradient step on randomly chosen point
        grad_step = differentiateSquaredLoss(X[random_index],Y[random_index],w)
        w_new = w - step_size*grad_step
        norm_evolution.append(np.linalg.norm(grad_step))

        if abs(objective_fn(w_new) - objective_fn(w)) < convergence: 
            converged = True
        
        # keep track of how the guess and function change as we run gradient descent  
        guess_evolution.append(w)
        fxn_evolution.append(objective_fn(w))

        # reset w to the newest version
        w = w_new
        # update step size to satisfy convergence criterion
        step_size = (step_size + num_iters)**0.75

    
    guess_evolution.append(w)
    fxn_evolution.append(objective_fn(w))
    norm_evolution.append(np.linalg.norm(grad_step))
    
    best_guess = w
    best_value = objective_fn(w)
    
    return best_guess, best_value, guess_evolution, fxn_evolution, norm_evolution

