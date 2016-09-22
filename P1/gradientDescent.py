import pylab as pl

def getData():
    
    # load the fitting data for X and y and return as elements of a tuple
    # X is a 100 by 10 matrix and y is a vector of length 100
    # Each corresponding row for X and y represents a single data sample

    X = pl.loadtxt('fittingdatap1_x.txt')
    y = pl.loadtxt('fittingdatap1_y.txt')

    return (X,y) 


'''
Implements batch gradient descent 

@param: objective_fn - 
@param: gradient_fn
@param: initial guess - 
@param: step size - 
@param: convergence criterion - threshold 

@return 
'''